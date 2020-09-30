# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch.nn.functional as F
import egg.core as core
from egg.zoo.population_signal_game.features import ImageNetFeat, ImagenetLoader
from egg.zoo.population_signal_game.archs import InformedSender,InformedSenderMultiHead, MyReceiver, Receiver

import wandb

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', default='', help='data root folder')
    # 2-agents specific parameters
    parser.add_argument('--tau_s', type=float, default=10.0,
                        help='Sender Gibbs temperature')
    parser.add_argument('--game_size', type=int, default=2,
                        help='Number of images seen by an agent')
    parser.add_argument('--pop_size', type=int, default=1,
                        help='Number of pop of one type')
    parser.add_argument('--same', type=int, default=0,
                        help='Use same concepts')
    parser.add_argument('--embedding_size', type=int, default=50,
                        help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=20,
                        help='hidden size (number of filters informed sender)')
    parser.add_argument('--batches_per_epoch', type=int, default=100,
                        help='Batches in a single training/validation epoch')
    parser.add_argument('--inf_rec', type=int, default=0,
                        help='Use informed receiver')
    parser.add_argument('--mode', type=str, default='rf',
                        help='Training mode: Gumbel-Softmax (gs) or Reinforce (rf). Default: rf.')
    parser.add_argument('--pop_mode', type=int, default=0,
                        help='0:simple, 1:crit aux, 2: crit baseline')
    parser.add_argument('--seed', type=int, default=0,
                        help='0, 1, 2, ...')
    parser.add_argument('--gs_tau', type=float, default=1.0,
                        help='GS temperature')
    parser.add_argument('--multi_head', type=int, default=0,
                        help='0, 1')
    parser.add_argument('--exp_prefix', type=str, default='',
                        help='blahh')
    opt = core.init(parser)
    assert opt.game_size >= 1

    return opt


def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    acc = (labels == receiver_output).float()
    # wandb.log({'acc':acc.mean()})
    return -acc, {'acc': acc.mean().item()}


def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels):
    """
    NLL loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float().mean()
    return nll, {'acc': acc}


def get_game(opt):
    feat_size = 4096
    pop = opts.pop_size
    sender_list = []
    receiver_list = []
    for i in range(pop):
        sender = InformedSender(opt.game_size, feat_size,
                                opt.embedding_size, opt.hidden_size, opt.vocab_size,
                                temp=opt.tau_s)
        receiver = Receiver(opt.game_size, feat_size,
                            opt.embedding_size, opt.vocab_size, reinforce=(opts.mode == 'rf'))
        if opts.mode == 'rf':
            sender = core.ReinforceWrapper(sender)
            receiver = core.ReinforceWrapper(receiver)
        sender.id = i
        receiver.id = i
        sender_list.append(sender)
        receiver_list.append(receiver)
    game = core.PopSymbolGameReinforce(
        sender_list, receiver_list, pop, loss, sender_entropy_coeff=0.01, receiver_entropy_coeff=0.01)
    elif opts.mode == 'gs':
        sender = core.GumbelSoftmaxWrapper(sender, temperature=opt.gs_tau)
        game = core.SymbolGameGS(sender, receiver, loss_nll)
    else:
        raise RuntimeError(f"Unknown training mode: {opts.mode}")

    return game

def get_my_game(opt):
    feat_size = 4096
    out_hidden_size = 20
    emb_size = 10
    pop = opts.pop_size
    sender_list = []
    receiver_list = []
    for i in range(pop):
        if not opts.multi_head:
            sender = InformedSender(opt.game_size, feat_size,
                                opt.embedding_size, opt.hidden_size, out_hidden_size,
                                temp=opt.tau_s)
        else:
            sender = InformedSenderMultiHead(opt.game_size, feat_size,
                                    opt.embedding_size, opt.hidden_size, out_hidden_size,
                                    temp=opt.tau_s)
        receiver = MyReceiver(opt.game_size, feat_size,
                            opt.embedding_size, out_hidden_size, reinforce=(opts.mode == 'rf'))

        if opts.mode == 'rf':
            sender = core.MyRnnSenderReinforce(sender, opt.vocab_size, emb_size, out_hidden_size,multi_head=opt.multi_head,
                                       cell="gru", max_len=opt.max_len)
            receiver = core.RnnReceiverReinforce(receiver, opt.vocab_size, emb_size,
                       out_hidden_size, cell="gru")
            receiver.multi_head = opts.multi_head==2
        elif opts.mode == 'gs':
            sender = core.GumbelSoftmaxWrapper(sender, temperature=opt.gs_tau)
        else:
            raise RuntimeError(f"Unknown training mode: {opts.mode}")
        sender.id = i
        receiver.id = i
        sender_list.append(sender)
        receiver_list.append(receiver)

    if opts.mode == 'rf':
        if opts.pop_mode == 0:
            game = core.PopSenderReceiverRnnReinforce(
                sender_list, receiver_list, pop, loss, sender_entropy_coeff=0.01, receiver_entropy_coeff=0.01)
        elif opts.pop_mode == 1:
            game = core.PopUncSenderReceiverRnnReinforce(
                sender_list, receiver_list, pop, loss, use_critic_baseline=False, sender_entropy_coeff=0.01, receiver_entropy_coeff=0.01)
        else:
            game = core.PopUncSenderReceiverRnnReinforce(
                sender_list, receiver_list, pop, loss, use_critic_baseline=True, sender_entropy_coeff=0.01, receiver_entropy_coeff=0.01)


    elif opts.mode == 'gs':
        game = core.PopSymbolGameGS(sender_list, receiver_list, pop, loss_nll)
    else:
        raise RuntimeError(f"Unknown training mode: {opts.mode}")

    return game


if __name__ == '__main__':
    opts = parse_arguments()
    wandb.init(project="referential-advice", name='{}-pop_seed-{}_size-{}_pop_mode-{}_multi_head-{}'.format(opts.exp_prefix, opts.seed, opts.pop_size, opts.pop_mode, opts.multi_head))
    wandb.config.exp_id = 'pop_size-{}_pop_mode-{}_multi_head-{}'.format(opts.pop_size, opts.pop_mode, opts.multi_head)

    data_folder = os.path.join(opts.root, "train/")
    dataset = ImageNetFeat(root=data_folder)

    train_loader = ImagenetLoader(dataset, batch_size=opts.batch_size, shuffle=True, opt=opts,
                                  batches_per_epoch=opts.batches_per_epoch, seed=None)
    validation_loader = ImagenetLoader(dataset, opt=opts, batch_size=opts.batch_size,
                                       batches_per_epoch=opts.batches_per_epoch,
                                       seed=7)
    game = get_game(opts)
    optimizer = core.build_optimizer(game.parameters())
    callback = None
    if opts.mode == 'gs':
        callbacks = [core.TemperatureUpdater(
            agent=game.sender, decay=0.9, minimum=0.1)]
    else:
        callbacks = []

    callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=validation_loader, callbacks=callbacks)

    trainer.train(n_epochs=opts.n_epochs)

    core.close()
