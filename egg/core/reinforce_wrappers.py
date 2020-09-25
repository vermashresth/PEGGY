# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .rnn import RnnEncoder
from .transformer import TransformerEncoder, TransformerDecoder
from .util import find_lengths
from .baselines import MeanBaseline, NNBaseline
from Levenshtein import distance as ld

import wandb

def cal_batch_ld(t1, t2):
    ar1 = t1.detach().cpu().numpy()
    ar2 = t2.detach().cpu().numpy()
    ar1_s = np.apply_along_axis(lambda x: ''.join(map(str,x)),1,ar1)
    ar2_s = np.apply_along_axis(lambda x: ''.join(map(str,x)),1,ar2)

    tot = np.vstack([ar1_s, ar2_s])
    batch_ld = np.apply_along_axis(lambda x: ld(x[0],x[1]), 0, tot)
    # mean_ld = np.mean(batch_ld)
    return batch_ld

class ReinforceWrapper(nn.Module):
    """
    Reinforce Wrapper for an agent. Assumes that the during the forward,
    the wrapped agent returns log-probabilities over the potential outputs. During training, the wrapper
    transforms them into a tuple of (sample from the multinomial, log-prob of the sample, entropy for the multinomial).
    Eval-time the sample is replaced with argmax.

    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = ReinforceWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4])
    >>> (log_prob < 0).all().item()
    1
    >>> (entropy > 0).all().item()
    1
    """
    def __init__(self, agent):
        super(ReinforceWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)

        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = logits.argmax(dim=1)
        log_prob = distr.log_prob(sample)

        return sample, log_prob, entropy


class ReinforceDeterministicWrapper(nn.Module):
    """
    Simple wrapper that makes a deterministic agent (without sampling) compatible with Reinforce-based game, by
    adding zero log-probability and entropy values to the output. No sampling is run on top of the wrapped agent,
    it is passed as is.
    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = ReinforceDeterministicWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4, 3])
    >>> (log_prob == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    """
    def __init__(self, agent):
        super(ReinforceDeterministicWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        out = self.agent(*args, **kwargs)

        return out, torch.zeros(1).to(out.device), torch.zeros(1).to(out.device)


class SymbolGameReinforce(nn.Module):
    """
    A single-symbol Sender/Receiver game implemented with Reinforce.
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0, baseline_type=MeanBaseline):
        """
        :param sender: Sender agent. On forward, returns a tuple of (message, log-prob of the message, entropy).
        :param receiver: Receiver agent. On forward, accepts a message and the dedicated receiver input. Returns
            a tuple of (output, log-probs, entropy).
        :param loss: The loss function that accepts:
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs the end-to-end loss. Can be non-differentiable; if it is differentiable, this will be leveraged
        :param sender_entropy_coeff: The entropy regularization coefficient for Sender
        :param receiver_entropy_coeff: The entropy regularizatino coefficient for Receiver
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        """
        super(SymbolGameReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.sender_entropy_coeff = sender_entropy_coeff

        self.baseline = baseline_type()

    def forward(self, sender_input, labels, receiver_input=None):
        message, sender_log_prob, sender_entropy = self.sender(sender_input)
        receiver_output, receiver_log_prob, receiver_entropy = self.receiver(message, receiver_input)

        loss, rest_info = self.loss(sender_input, message, receiver_input, receiver_output, labels)
        policy_loss = ((loss.detach() - self.baseline.predict(loss.detach())) * (sender_log_prob + receiver_log_prob)).mean()
        entropy_loss = -(sender_entropy.mean() * self.sender_entropy_coeff + receiver_entropy.mean() * self.receiver_entropy_coeff)

        if self.training:
            self.baseline.update(loss.detach())

        full_loss = policy_loss + entropy_loss + loss.mean()

        for k, v in rest_info.items():
            if hasattr(v, 'mean'):
                rest_info[k] = v.mean().item()

        rest_info['baseline'] = self.baseline.predict(loss.detach()).mean()
        rest_info['loss'] = loss.mean().item()
        rest_info['sender_entropy'] = sender_entropy.mean()
        rest_info['receiver_entropy'] = receiver_entropy.mean()

        return full_loss, rest_info

class MyRnnSenderReinforce(nn.Module):
    """
    Reinforce Wrapper for Sender in variable-length message game. Assumes that during the forward,
    the wrapped agent returns the initial hidden state for a RNN cell. This cell is the unrolled by the wrapper.
    During training, the wrapper samples from the cell, getting the output message. Evaluation-time, the sampling
    is replaced by argmax.

    >>> agent = nn.Linear(10, 3)
    >>> agent = RnnSenderReinforce(agent, vocab_size=5, embed_dim=5, hidden_size=3, max_len=10, cell='lstm', force_eos=False)
    >>> input = torch.FloatTensor(16, 10).uniform_(-0.1, 0.1)
    >>> message, logprob, entropy = agent(input)
    >>> message.size()
    torch.Size([16, 10])
    >>> (entropy > 0).all().item()
    1
    >>> message.size()  # batch size x max_len
    torch.Size([16, 10])
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, max_len, multi_head=False, num_layers=1, cell='rnn', force_eos=True):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        :param force_eos: if set to True, each message is extended by an EOS symbol. To ensure that no message goes
        beyond `max_len`, Sender only generates `max_len - 1` symbols from an RNN cell and appends EOS.
        """
        super(MyRnnSenderReinforce, self).__init__()
        self.agent = agent

        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            assert self.max_len > 1, "Cannot force eos when max_len is below 1"
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None
        self.multi_head = multi_head
        self.unc_threshold = 2
        self.advice_info = None
        cell = cell.lower()
        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)])
        self.advice_mode(False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def set_manager(self, manager):
        self.manager = manager

    def advice_mode(self, flag):
        self.give_advice = flag
        self.training = not flag
        # self.agent.advice_mode = flag
        if flag:
            self.eval()
        else:
            self.train()

    def forward(self, x):
        sequence_list, logits_list, entropy_list = [], [], []
        agent_outs = self.agent(x)
        if not self.multi_head:
            agent_outs = [agent_outs]
        distributions = []
        for head_id, output in enumerate(agent_outs):
            prev_hidden = [output]
            prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

            prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

            input = torch.stack([self.sos_embedding] * x.size(1))

            sequence = []
            logits = []
            entropy = []
            step_distr = []
            for step in range(self.max_len):
                for i, layer in enumerate(self.cells):
                    if isinstance(layer, nn.LSTMCell):
                        h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                        prev_c[i] = c_t
                    else:
                        h_t = layer(input, prev_hidden[i])
                    prev_hidden[i] = h_t
                    input = h_t

                step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
                distr = Categorical(logits=step_logits)
                entropy.append(distr.entropy())
                step_distr.append(distr)

                if self.training or self.give_advice:
                    x_out = distr.sample()
                else:
                    x_out = step_logits.argmax(dim=1)
                logits.append(distr.log_prob(x_out))

                input = self.embedding(x_out)
                sequence.append(x_out)

            distributions.append(step_distr)

            sequence = torch.stack(sequence).permute(1, 0)
            logits = torch.stack(logits).permute(1, 0)
            entropy = torch.stack(entropy).permute(1, 0)

            if self.force_eos:
                zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

                sequence = torch.cat([sequence, zeros.long()], dim=1)
                logits = torch.cat([logits, zeros], dim=1)
                entropy = torch.cat([entropy, zeros], dim=1)
            sequence_list.append(sequence)
            logits_list.append(logits)
            entropy_list.append(entropy)
        final_id = np.random.choice(range(len(agent_outs)))
        # print(sequence_list[0], sequence_list[1])
        if self.multi_head:
            self.unc_batch = cal_batch_ld(sequence_list[0], sequence_list[1])
            self.unc = np.mean(self.unc_batch)
            if self.training:
                wandb.log({'message unc':self.unc})
            else:
                wandb.log({'message argmax unc':self.unc})
            if self.multi_head >1:
                if not self.give_advice:
                    if self.advice_info is None:
                        output = self.manager.get_advice(x, self.id)
                        do_ask_advice = self.unc_batch > self.unc_threshold
                        self.advice_info = [output, do_ask_advice]
                    else:
                        output, do_ask_advice = self.advice_info
                    adviced_seq = torch.tensor(output[1]) # batch of seq len , 32x5
                    unc_of_adviced = torch.Tensor(output[0])
                    step_distr = distributions[final_id]
                    for step in range(self.max_len):
                        step_logits_for_adviced = step_distr[step].log_prob(adviced_seq[:, step].cuda()) # batch size
                        for idx, flag in enumerate(do_ask_advice):
                            if flag and unc_of_adviced[idx]!=100:
                                logits_list[final_id][idx] = step_logits_for_adviced[idx]
                                sequence_list[final_id][idx] = adviced_seq[:, step][idx]
                    return sequence_list[final_id], logits_list[final_id], entropy_list[final_id]
                else:
                    ## Giving advice
                    return self.unc_batch, sequence_list[final_id], logits_list[final_id], entropy_list[final_id]
        return sequence_list[final_id], logits_list[final_id], entropy_list[final_id]

class RnnSenderReinforce(nn.Module):
    """
    Reinforce Wrapper for Sender in variable-length message game. Assumes that during the forward,
    the wrapped agent returns the initial hidden state for a RNN cell. This cell is the unrolled by the wrapper.
    During training, the wrapper samples from the cell, getting the output message. Evaluation-time, the sampling
    is replaced by argmax.

    >>> agent = nn.Linear(10, 3)
    >>> agent = RnnSenderReinforce(agent, vocab_size=5, embed_dim=5, hidden_size=3, max_len=10, cell='lstm', force_eos=False)
    >>> input = torch.FloatTensor(16, 10).uniform_(-0.1, 0.1)
    >>> message, logprob, entropy = agent(input)
    >>> message.size()
    torch.Size([16, 10])
    >>> (entropy > 0).all().item()
    1
    >>> message.size()  # batch size x max_len
    torch.Size([16, 10])
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, max_len, num_layers=1, cell='rnn', force_eos=True):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        :param force_eos: if set to True, each message is extended by an EOS symbol. To ensure that no message goes
        beyond `max_len`, Sender only generates `max_len - 1` symbols from an RNN cell and appends EOS.
        """
        super(RnnSenderReinforce, self).__init__()
        self.agent = agent

        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            assert self.max_len > 1, "Cannot force eos when max_len is below 1"
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None

        cell = cell.lower()
        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x):
        prev_hidden = [self.agent(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))

            input = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy


class RnnReceiverReinforce(nn.Module):
    """
    Reinforce Wrapper for Receiver in variable-length message game. The wrapper logic feeds the message into the cell
    and calls the wrapped agent on the hidden state vector for the step that either corresponds to the EOS input to the
    input that reaches the maximal length of the sequence.
    This output is assumed to be the tuple of (output, logprob, entropy).
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell='rnn', num_layers=1):
        super(RnnReceiverReinforce, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)
        sample, logits, entropy = self.agent(encoded, input)

        return sample, logits, entropy


class RnnReceiverDeterministic(nn.Module):
    """
    Reinforce Wrapper for a deterministic Receiver in variable-length message game. The wrapper logic feeds the message
    into the cell and calls the wrapped agent with the hidden state that either corresponds to the end-of-sequence
    term or to the end of the sequence. The wrapper extends it with zero-valued log-prob and entropy tensors so that
    the agent becomes compatible with the SenderReceiverRnnReinforce game.

    As the wrapped agent does not sample, it has to be trained via regular back-propagation. This requires that both the
    the agent's output and  loss function and the wrapped agent are differentiable.

    >>> class Agent(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> agent = RnnReceiverDeterministic(Agent(), vocab_size=10, embed_dim=10, hidden_size=5)
    >>> message = torch.zeros((16, 10)).long().random_(0, 10)  # batch of 16, 10 symbol length
    >>> output, logits, entropy = agent(message)
    >>> (logits == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    >>> output.size()
    torch.Size([16, 3])
    """

    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell='rnn', num_layers=1):
        super(RnnReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)
        agent_output = self.agent(encoded, input)

        logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
        entropy = logits

        return agent_output, logits, entropy

class MMRnnReceiverDeterministic(nn.Module):
    """
    Reinforce Wrapper for a deterministic Receiver in variable-length message game. The wrapper logic feeds the message
    into the cell and calls the wrapped agent with the hidden state that either corresponds to the end-of-sequence
    term or to the end of the sequence. The wrapper extends it with zero-valued log-prob and entropy tensors so that
    the agent becomes compatible with the SenderReceiverRnnReinforce game.

    As the wrapped agent does not sample, it has to be trained via regular back-propagation. This requires that both the
    the agent's output and  loss function and the wrapped agent are differentiable.

    >>> class Agent(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> agent = RnnReceiverDeterministic(Agent(), vocab_size=10, embed_dim=10, hidden_size=5)
    >>> message = torch.zeros((16, 10)).long().random_(0, 10)  # batch of 16, 10 symbol length
    >>> output, logits, entropy = agent(message)
    >>> (logits == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    >>> output.size()
    torch.Size([16, 3])
    """

    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell='rnn', num_layers=1):
        super(MMRnnReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder1 = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)
        self.encoder2 = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):
        message1, message2 = torch.split(message, 2, dim=1)
        encoded1 = self.encoder1(message1)
        encoded2 = self.encoder2(message2)
        encoded = torch.cat([encoded1, encoded2], dim=1)
        agent_output = self.agent(encoded, input)

        logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
        entropy = logits

        return agent_output, logits, entropy


class SenderReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/Receiver game with training done via Reinforce. Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.

    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce the variance of the
    gradient estimate.

    >>> sender = nn.Linear(3, 10)
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    ...     return F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1), {'aux': 5.0}

    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((16, 3)).normal_()
    >>> optimized_loss, aux_info = game(input, labels=None)
    >>> sorted(list(aux_info.keys()))  # returns some debug info, such as entropies of the agents, message length etc
    ['aux', 'loss', 'mean_length', 'original_loss', 'receiver_entropy', 'sender_entropy']
    >>> aux_info['aux']
    5.0
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff, receiver_entropy_coeff,
                 length_cost=0.0, baseline_type=MeanBaseline):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        """
        super(SenderReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost
        self.baselines = defaultdict(baseline_type)

    def forward(self, sender_input, labels, receiver_input=None):
        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = find_lengths(message)
        receiver_output, log_prob_r, entropy_r = self.receiver(message, receiver_input, message_lengths)

        loss, rest = self.loss(sender_input, message, receiver_input, receiver_output, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = ((length_loss - self.baselines['length'].predict(length_loss)) * effective_log_prob_s).mean()
        policy_loss = ((loss.detach() - self.baselines['loss'].predict(loss.detach())) * log_prob).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy
        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.baselines['loss'].update(loss)
            self.baselines['length'].update(length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

class PopSenderReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/Receiver game with training done via Reinforce. Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.

    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce the variance of the
    gradient estimate.

    >>> sender = nn.Linear(3, 10)
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    ...     return F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1), {'aux': 5.0}

    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((16, 3)).normal_()
    >>> optimized_loss, aux_info = game(input, labels=None)
    >>> sorted(list(aux_info.keys()))  # returns some debug info, such as entropies of the agents, message length etc
    ['aux', 'loss', 'mean_length', 'original_loss', 'receiver_entropy', 'sender_entropy']
    >>> aux_info['aux']
    5.0
    """
    def __init__(self, sender_list, receiver_list, pop, loss, sender_entropy_coeff, receiver_entropy_coeff,
                 length_cost=0.0, baseline_type=MeanBaseline):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        """
        super(PopSenderReceiverRnnReinforce, self).__init__()
        self.sender_list = nn.ModuleList(sender_list)
        self.receiver_list = nn.ModuleList(receiver_list)
        self.pop = pop
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost

        self.baselines = defaultdict(baseline_type)
        self.s_advice_manager = AdviceManager(sender_list)

        self.learn_advice_iters = 5
        for sender in self.sender_list:
            sender.set_manager(self.s_advice_manager)

    def forward(self, sender_input, labels, receiver_input=None):
        index = np.random.choice(range(self.pop))

        self.sender = self.sender_list[index]
        self.receiver = self.receiver_list[index]

        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = find_lengths(message)
        receiver_output, log_prob_r, entropy_r = self.receiver(message, receiver_input, message_lengths)

        loss, rest = self.loss(sender_input, message, receiver_input, receiver_output, labels)
        s_loss = loss.clone()
        tot_loss = loss.clone()
        for i in range(self.learn_advice_iters):
            n_message, n_log_prob_s, n_entropy_s = self.sender(sender_input)
            do_ask_advice = torch.Tensor(self.sender.advice_info[1])==True
            f_message, f_log_prob_s, f_entropy_s = n_message[do_ask_advice], n_log_prob_s[do_ask_advice], n_entropy_s[do_ask_advice]
            f_loss = s_loss[do_ask_advice]
            message = torch.cat([message, f_message])
            log_prob_s = torch.cat([log_prob_s, f_log_prob_s])
            entropy_s = torch.cat([entropy_s, f_entropy_s])
            tot_loss = torch.cat([tot_loss, f_loss])

        self.sender.advice_info = None
        message_lengths = find_lengths(message)
        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_s[:, 0])

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_s[:, 0])

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        # log_prob = effective_log_prob_s + log_prob_r

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = ((length_loss - self.baselines['length'].predict(length_loss)) * effective_log_prob_s).mean()
        policy_loss_s = ((tot_loss.detach() - self.baselines['s_loss'].predict(tot_loss.detach())) * effective_log_prob_s).mean()
        policy_loss_r = ((loss.detach() - self.baselines['loss'].predict(loss.detach())) * log_prob_r).mean()
        policy_loss = policy_loss_r+policy_loss_s
        optimized_loss = policy_length_loss + policy_loss - weighted_entropy
        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()
        wandb.log({'Critic losses':0.0, 'policy_loss':policy_loss.mean()})

        if self.training:
            self.baselines['s_loss'].update(tot_loss)
            self.baselines['loss'].update(loss)
            self.baselines['length'].update(length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

class AdviceManager():
    def __init__(self, agent_list, unc_threshold=2):
        self.agent_list = agent_list
        self.unc_threshold = unc_threshold
    def get_advice(self, state, my_id):
        batch_size = state.size(1)
        min_unc = [100 for _ in range(batch_size)]
        f_message = [ [0 for __ in range(5)] for _ in range(batch_size)]
        uncs, messages = [], []
        for id, agent in enumerate(self.agent_list):
            if id!=my_id:
                agent.advice_mode(True)
                unc, message, _, _ = agent(state)
                agent.advice_mode(False)

                uncs.append(unc)
                messages.append(message.detach().cpu().numpy())

        for id in range(len(uncs)):
            for bt in range(batch_size):
                if uncs[id][bt] < self.unc_threshold and uncs[id][bt] < min_unc[bt]:
                    min_unc[bt] = uncs[id][bt]
                    f_message[bt] = messages[id][bt]
        return min_unc, f_message, _, _

class PopUncSenderReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/Receiver game with training done via Reinforce. Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.

    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce the variance of the
    gradient estimate.

    >>> sender = nn.Linear(3, 10)
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    ...     return F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1), {'aux': 5.0}

    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((16, 3)).normal_()
    >>> optimized_loss, aux_info = game(input, labels=None)
    >>> sorted(list(aux_info.keys()))  # returns some debug info, such as entropies of the agents, message length etc
    ['aux', 'loss', 'mean_length', 'original_loss', 'receiver_entropy', 'sender_entropy']
    >>> aux_info['aux']
    5.0
    """
    def __init__(self, sender_list, receiver_list, pop, loss, sender_entropy_coeff, receiver_entropy_coeff,
                 use_critic_baseline=False, length_cost=0.0, baseline_type=MeanBaseline, critic_type=NNBaseline):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        """
        super(PopUncSenderReceiverRnnReinforce, self).__init__()
        self.sender_list = nn.ModuleList(sender_list)
        self.receiver_list = nn.ModuleList(receiver_list)
        self.pop = pop
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost
        self.use_critic_baseline = use_critic_baseline
        game_size = self.sender_list[0].agent.game_size
        embedding_size = self.sender_list[0].agent.embedding_size
        hidden_size = self.sender_list[0].agent.hidden_size
        self.mean_baselines = [defaultdict(baseline_type) for _ in range(pop)]
        self.s_critics = [critic_type(embedding_size, hidden_size) for _ in range(pop)]
        self.r_critics = [critic_type(game_size, 2) for _ in range(pop)]

        self.s_advice_manager = AdviceManager(sender_list)
        for sender in self.sender_list:
            sender.set_manager(self.s_advice_manager)

    def forward(self, sender_input, labels, receiver_input=None):
        index = np.random.choice(range(self.pop))

        self.sender = self.sender_list[index]
        self.receiver = self.receiver_list[index]
        self.s_critic = self.s_critics[index]
        self.r_critic = self.r_critics[index]
        self.mean_baseline = self.mean_baselines[index]

        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = find_lengths(message)
        receiver_output, log_prob_r, entropy_r = self.receiver(message, receiver_input, message_lengths)

        loss, rest = self.loss(sender_input, message, receiver_input, receiver_output, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = ((length_loss - self.mean_baseline['length'].predict(length_loss)) * effective_log_prob_s).mean()
        if not self.use_critic_baseline:
            policy_loss = ((loss.detach() - self.mean_baseline['loss'].predict(loss.detach())) * log_prob).mean()
        else:
            policy_loss_s = ((loss.detach() + self.s_critic.predict(self.sender.agent.return_final_encodings())) * effective_log_prob_s).mean()
            policy_loss_r = ((loss.detach() + self.r_critic.predict(self.receiver.agent.return_final_encodings())) * log_prob_r).mean()

        sc = self.s_critic.predict(self.sender.agent.return_final_encodings())
        _ = self.r_critic.predict(self.receiver.agent.return_final_encodings())
        critic_loss_s = self.s_critic.get_loss(loss)
        critic_loss_r = self.r_critic.get_loss(loss)

        if not self.use_critic_baseline:
            optimized_loss = policy_length_loss + policy_loss - weighted_entropy
        else:
            optimized_loss = policy_length_loss + policy_loss_s + policy_loss_r - weighted_entropy

        critic_losses = critic_loss_s + critic_loss_r
        optimized_loss += critic_losses
        if not self.use_critic_baseline:
            wandb.log({'Critic losses':critic_losses.mean(), 'policy_loss':policy_loss.mean()})
            wandb.log({'Critic predict s': sc.mean(), 'ac_loss':loss.mean(), 'critic loss s':critic_loss_s })
        else:
            wandb.log({'Critic losses':critic_losses.mean(), 'policy_loss':(policy_loss_s + policy_loss_r).mean()})
            wandb.log({'Critic predict s': sc.mean(), 'ac_loss':loss.mean(), 'critic loss s':critic_loss_s })

        # optimized_loss += critic_loss_s

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            if not self.use_critic_baseline:
                self.mean_baseline['loss'].update(loss)
            self.mean_baseline['length'].update(length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

class TransformerReceiverDeterministic(nn.Module):
    def __init__(self, agent, vocab_size, max_len, embed_dim, num_heads, hidden_size, num_layers, positional_emb=True,
                causal=True):
        super(TransformerReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder = TransformerEncoder(vocab_size=vocab_size,
                                          max_len=max_len,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          num_layers=num_layers,
                                          hidden_size=hidden_size,
                                          positional_embedding=positional_emb,
                                          causal=causal)

    def forward(self, message, input=None, lengths=None):
        if lengths is None:
            lengths = find_lengths(message)

        transformed = self.encoder(message, lengths)
        agent_output = self.agent(transformed, input)

        logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
        entropy = logits

        return agent_output, logits, entropy


class TransformerSenderReinforce(nn.Module):
    def __init__(self, agent, vocab_size, embed_dim, max_len, num_layers, num_heads, hidden_size,
                 generate_style='standard', causal=True, force_eos=True):
        """
        :param agent: the agent to be wrapped, returns the "encoder" state vector, which is the unrolled into a message
        :param vocab_size: vocab size of the message
        :param embed_dim: embedding dimensions
        :param max_len: maximal length of the message (including <eos>)
        :param num_layers: number of transformer layers
        :param num_heads: number of attention heads
        :param hidden_size: size of the FFN layers
        :param causal: whether embedding of a particular symbol should only depend on the symbols to the left
        :param generate_style: Two alternatives: 'standard' and 'in-place'. Suppose we are generating 4th symbol,
            after three symbols [s1 s2 s3] were generated.
            Then,
            'standard': [s1 s2 s3] -> embeddings [[e1] [e2] [e3]] -> (s4 = argmax(linear(e3)))
            'in-place': [s1 s2 s3] -> [s1 s2 s3 <need-symbol>] -> embeddings [[e1] [e2] [e3] [e4]] -> (s4 = argmax(linear(e4)))
        :param force_eos: <eos> added to the end of each sequence
        """
        super(TransformerSenderReinforce, self).__init__()
        self.agent = agent

        self.force_eos = force_eos
        assert generate_style in ['standard', 'in-place']
        self.generate_style = generate_style
        self.causal = causal

        self.max_len = max_len

        if force_eos:
            self.max_len -= 1

        self.transformer = TransformerDecoder(embed_dim=embed_dim,
                                              max_len=max_len, num_layers=num_layers,
                                              num_heads=num_heads, hidden_size=hidden_size)

        self.embedding_to_vocab = nn.Linear(embed_dim, vocab_size)

        self.special_symbol_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embed_tokens = torch.nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_dim ** -0.5)
        self.embed_scale = math.sqrt(embed_dim)

    def generate_standard(self, encoder_state):
        batch_size = encoder_state.size(0)
        device = encoder_state.device

        sequence = []
        logits = []
        entropy = []

        special_symbol = self.special_symbol_embedding.expand(batch_size, -1).unsqueeze(1).to(device)
        input = special_symbol

        for step in range(self.max_len):
            if self.causal:
                attn_mask = torch.triu(torch.ones(step+1, step+1).byte(), diagonal=1).to(device)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float('-inf'))
            else:
                attn_mask = None
            output = self.transformer(embedded_input=input, encoder_out=encoder_state, attn_mask=attn_mask)
            step_logits = F.log_softmax(self.embedding_to_vocab(output[:, -1, :]), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())
            if self.training:
                symbols = distr.sample()
            else:
                symbols = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(symbols))
            sequence.append(symbols)

            new_embedding = self.embed_tokens(symbols) * self.embed_scale
            input = torch.cat([input, new_embedding.unsqueeze(dim=1)], dim=1)

        return sequence, logits, entropy

    def generate_inplace(self, encoder_state):
        batch_size = encoder_state.size(0)
        device = encoder_state.device

        sequence = []
        logits = []
        entropy = []

        special_symbol = self.special_symbol_embedding.expand(batch_size, -1).unsqueeze(1).to(encoder_state.device)
        output = []
        for step in range(self.max_len):
            input = torch.cat(output + [special_symbol], dim=1)
            if self.causal:
                attn_mask = torch.triu(torch.ones(step+1, step+1).byte(), diagonal=1).to(device)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float('-inf'))
            else:
                attn_mask = None

            embedded = self.transformer(embedded_input=input, encoder_out=encoder_state, attn_mask=attn_mask)
            step_logits = F.log_softmax(self.embedding_to_vocab(embedded[:, -1, :]), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())
            if self.training:
                symbols = distr.sample()
            else:
                symbols = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(symbols))
            sequence.append(symbols)

            new_embedding = self.embed_tokens(symbols) * self.embed_scale
            output.append(new_embedding.unsqueeze(dim=1))

        return sequence, logits, entropy

    def forward(self, x):
        encoder_state = self.agent(x)

        if self.generate_style == 'standard':
            sequence, logits, entropy = self.generate_standard(encoder_state)
        elif self.generate_style == 'in-place':
            sequence, logits, entropy = self.generate_inplace(encoder_state)
        else:
            assert False, 'Unknown generate style'

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy
