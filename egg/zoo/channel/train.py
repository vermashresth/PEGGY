# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.channel.features import OneHotLoader, UniformLoader
from egg.zoo.channel.archs import Sender, Receiver


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=10,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')
    parser.add_argument('--dim_dataset', type=int, default=10240,
                        help='Dim of constructing the data (default: 10240)')
    parser.add_argument('--force_eos', type=int, default=0,
                        help='Force EOS at the end of the messages (default: 0)')

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--receiver_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--sender_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')

    parser.add_argument('--sender_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument('--sender_cell', type=str, default='rnn',
                        help='Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)')
    parser.add_argument('--receiver_cell', type=str, default='rnn',
                        help='Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)')


    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Sender (default: 1e-1)')
    parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Receiver (default: 1e-1)')

    parser.add_argument('--temperature', type=float, default=1.0,
                        help="GS temperature for the sender (default: 1.0)")
    parser.add_argument('--mode', type=str, default='rf',
                        help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training {rf, gs}"
                             "(default: rf)")
    parser.add_argument('--probs', type=str, default='uniform',
                        help="Prior distribution over the concepts (default: uniform)")
    parser.add_argument('--length_cost', type=float, default=0.0,
                        help="Penalty for the message length, each symbol would before <EOS> would be "
                             "penalized by this cost (default: 0.0)")
    parser.add_argument('--name', type=str, default='model',
                        help="Name for your checkpoint (default: model)")
    parser.add_argument('--early_stopping_thr', type=float, default=0.9999,
                        help="Early stopping threshold on accuracy (defautl: 0.9999)")

    args = core.init(parser, params)
    return args


def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")
    return loss, {'acc': acc}


def dump(game, n_features, device, gs_mode):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
        core.dump_sender_receiver(game, dataset, gs=gs_mode, device=device, variable_length=True)

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    for sender_input, message, receiver_output in zip(sender_inputs, messages, receiver_outputs):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(f'Mean accuracy wrt uniform distribution is {unif_acc}')
    print(f'Mean accuracy wrt powerlaw distribution is {powerlaw_acc}')
    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))


def main(params):
    opts = get_params(params)
    print(opts, flush=True)
    device = opts.device

    force_eos = opts.force_eos == 1

    if opts.probs == 'uniform':
        probs = np.ones(opts.n_features)
    elif opts.probs == 'powerlaw':
        probs = 1 / np.arange(1, opts.n_features+1, dtype=np.float32)
    else:
        probs = np.array([float(x) for x in opts.probs.split(',')], dtype=np.float32)
    probs /= probs.sum()

    print('the probs are: ', probs, flush=True)

    train_loader = OneHotLoader(n_features=opts.n_features, batch_size=opts.batch_size,
                                batches_per_epoch=opts.batches_per_epoch, probs=probs)

    # single batches with 1s on the diag
    test_loader = UniformLoader(opts.n_features)

    sender = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
    receiver = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)

    if opts.mode.lower() == 'rf':
        sender = core.RnnSenderReinforce(sender,
                                   opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                   cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                   force_eos=force_eos)
        receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, opts.receiver_embedding,
                                                 opts.receiver_hidden, cell=opts.receiver_cell,
                                                 num_layers=opts.receiver_num_layers)

        game = core.SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff,
                                               receiver_entropy_coeff=opts.receiver_entropy_coeff,
                                               length_cost=opts.length_cost)

        callback = None
    elif opts.mode.lower() == 'gs':
        sender = core.RnnSenderGS(sender, opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                   cell=opts.sender_cell, max_len=opts.max_len, temperature=opts.temperature,
                                   force_eos=force_eos)
        receiver = core.RnnReceiverGS(receiver, opts.vocab_size, opts.receiver_embedding,
                    opts.receiver_hidden, cell=opts.receiver_cell)

        game = core.SenderReceiverRnnGS(sender, receiver, loss, opts.length_cost)
        callback = sender.update_temp(0.95, 0.1)
    else:
        raise NotImplementedError(f'Unknown training mode, {opts.mode}')

    optimizer = torch.optim.Adam([
        {'params': game.sender.parameters(), 'lr': opts.lr},
        {'params': game.receiver.parameters(), 'lr': opts.lr}
    ])

    early_stopper = EarlyStopperAccuracy(opts.early_stopping_thr)

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader, epoch_callback=callback, early_stopping=early_stopper)

    trainer.train(n_epochs=opts.n_epochs)
    if opts.checkpoint_dir:
        trainer.save_checkpoint(name=f'{opts.name}_vocab{opts.vocab_size}_rs{opts.random_seed}_lr{opts.lr}_shid{opts.sender_hidden}_rhid{opts.receiver_hidden}_sentr{opts.sender_entropy_coeff}_reg{opts.length_cost}_max_len{opts.max_len}')

    dump(trainer.game, opts.n_features, device, opts.mode.lower() == 'gs')
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])