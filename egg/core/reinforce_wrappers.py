# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict


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
    def __init__(self, sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0):
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
        """
        super(SymbolGameReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.sender_entropy_coeff = sender_entropy_coeff

        self.mean_baseline = 0.0
        self.n_points = 0.0

    def forward(self, sender_input, labels, receiver_input=None):
        message, sender_log_prob, sender_entropy = self.sender(sender_input)
        receiver_output, receiver_log_prob, receiver_entropy = self.receiver(message, receiver_input)

        loss, rest_info = self.loss(sender_input, message, receiver_input, receiver_output, labels)
        policy_loss = ((loss.detach() - self.mean_baseline) * (sender_log_prob + receiver_log_prob)).mean()
        entropy_loss = -(sender_entropy.mean() * self.sender_entropy_coeff + receiver_entropy.mean() * self.receiver_entropy_coeff)

        if self.training:
            self.n_points += 1.0
            self.mean_baseline += (loss.detach().mean().item() -
                                   self.mean_baseline) / self.n_points

        full_loss = policy_loss + entropy_loss + loss.mean()

        rest_info['baseline'] = self.mean_baseline
        rest_info['loss'] = loss.mean().item()
        rest_info['sender_entropy'] = sender_entropy.mean()
        rest_info['receiver_entropy'] = receiver_entropy.mean()

        return full_loss, rest_info


def _find_lengths(messages):
    """
    >>> messages = torch.tensor([[1, 1, 0, 0, 0, 1], [1, 1, 1, 10, 100500, 5]])
    >>> lengths = _find_lengths(messages)
    >>> lengths
    tensor([3, 6])
    """
    max_k = messages.size(1)
    zero_mask = messages == 0
    # a bit involved logic, but it seems to be faster for large batches than slicing batch dimension and
    # querying torch.nonzero()
    # zero_mask contains ones on positions where 0 occur in the outputs, and 1 otherwise
    # zero_mask.cumsum(dim=1) would contain non-zeros on all positions after 0 occurred
    # zero_mask.cumsum(dim=1) > 0 would contain ones on all positions after 0 occurred
    # (zero_mask.cumsum(dim=1) > 0).sum(dim=1) equates to the number of steps that happened after 0 occured (including it)
    # max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1) is the number of steps before 0 took place

    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths


def _get_batch_permutation(lengths):
    """
    Returns a permutation and its reverse that turns `lengths` in a sorted
    list in descending order.
    >>> lengths = torch.tensor([4, 1, 0, 100])
    >>> permutation, inverse = _get_batch_permutation(lengths)
    >>> permutation
    tensor([3, 0, 1, 2])
    >>> rearranged = torch.index_select(lengths, 0, permutation)
    >>> rearranged
    tensor([100,   4,   1,   0])
    >>> torch.index_select(rearranged, 0, inverse)
    tensor([  4,   1,   0, 100])
    """
    _, rearrange = torch.sort(lengths, descending=True)
    inverse = torch.empty_like(rearrange)
    inverse.scatter_(0, rearrange,
                    torch.arange(0, rearrange.numel(), device=rearrange.device))
    return rearrange, inverse


class RnnSenderReinforce(nn.Module):
    """
    Reinforce Wrapper for Sender in variable-length message game. Assumes that during the forward,
    the wrapped agent returns the initial hidden state for a RNN cell. This cell is the unrolled by the wrapper.
    During training, the wrapper samples from the cell, getting the output message. Evaluation-time, the sampling
    is replaced by argmax.

    >>> agent = nn.Linear(10, 3)
    >>> agent = RnnSenderReinforce(agent, vocab_size=5, emb_dim=5, n_hidden=3, max_len=10, cell='lstm')
    >>> input = torch.FloatTensor(16, 10).uniform_(-0.1, 0.1)
    >>> message, logprob, entropy = agent(input)
    >>> message.size()
    torch.Size([16, 10])
    >>> (entropy > 0).all().item()
    1
    >>> message.size()  # batch size x max_len
    torch.Size([16, 10])
    """
    def __init__(self, agent, vocab_size, emb_dim, n_hidden, max_len, num_layers=1, cell='rnn', force_eos=False):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param emb_dim: the size of the embedding used to embed the output symbols
        :param n_hidden: the RNN cell's hidden state size
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
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(n_hidden, vocab_size)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(emb_dim))
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None

        cell = cell.lower()
        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList([
            cell_type(input_size=emb_dim, hidden_size=n_hidden) if i == 0 else \
            cell_type(input_size=n_hidden, hidden_size=n_hidden) for i in range(self.num_layers)])

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
    def __init__(self, agent, vocab_size, emb_dim, n_hidden, cell='rnn', num_layers=1):
        super(RnnReceiverReinforce, self).__init__()
        self.agent = agent

        self.cell = None
        cell = cell.lower()
        if cell == 'rnn':
            self.cell = nn.RNN(input_size=emb_dim, batch_first=True, hidden_size=n_hidden, num_layers=num_layers)
        elif cell == 'gru':
            self.cell = nn.GRU(input_size=emb_dim, batch_first=True, hidden_size=n_hidden, num_layers=num_layers)
        elif cell == 'lstm':
            self.cell = nn.LSTM(input_size=emb_dim, batch_first=True, hidden_size=n_hidden, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, message, input=None, lengths=None):
        emb = self.embedding(message)

        if lengths is None:
            lengths = _find_lengths(message)

        permutation, inverse = _get_batch_permutation(lengths)

        emb = torch.index_select(emb, 0, permutation)
        lengths = torch.index_select(lengths, 0, permutation)

        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)
        _, rnn_hidden = self.cell(packed)

        if isinstance(self.cell, nn.LSTM):
            rnn_hidden, _ = rnn_hidden

        sample, logits, entropy = self.agent(rnn_hidden[-1], input)

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
    >>> agent = RnnReceiverDeterministic(Agent(), vocab_size=10, emb_dim=10, n_hidden=5)
    >>> message = torch.zeros((16, 10)).long().random_(0, 10)  # batch of 16, 10 symbol length
    >>> output, logits, entropy = agent(message)
    >>> (logits == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    >>> output.size()
    torch.Size([16, 3])
    """

    def __init__(self, agent, vocab_size, emb_dim, n_hidden, cell='rnn', num_layers=1):
        super(RnnReceiverDeterministic, self).__init__()
        self.agent = agent

        self.cell = None
        cell = cell.lower()
        if cell == 'rnn':
            self.cell = nn.RNN(input_size=emb_dim, batch_first=True, hidden_size=n_hidden, num_layers=num_layers)
        elif cell == 'gru':
            self.cell = nn.GRU(input_size=emb_dim, batch_first=True, hidden_size=n_hidden, num_layers=num_layers)
        elif cell == 'lstm':
            self.cell = nn.LSTM(input_size=emb_dim, batch_first=True, hidden_size=n_hidden, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, message, input=None, lengths=None):
        emb = self.embedding(message)

        if lengths is None:
            lengths = _find_lengths(message)
        permutation, inverse = _get_batch_permutation(lengths)

        emb = torch.index_select(emb, 0, permutation)
        lengths = torch.index_select(lengths, 0, permutation)

        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)
        _, rnn_hidden = self.cell(packed)

        if isinstance(self.cell, nn.LSTM):
            rnn_hidden, _ = rnn_hidden

        agent_output = self.agent(rnn_hidden[-1], input)
        agent_output = torch.index_select(agent_output, 0, inverse)

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
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, emb_dim=5, n_hidden=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, emb_dim=10, n_hidden=5)
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
                 length_cost=0.0):
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
        """
        super(SenderReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)

    def forward(self, sender_input, labels, receiver_input=None):
        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = _find_lengths(message)
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

        policy_length_loss = ((length_loss.float() - self.mean_baseline['length']) * effective_log_prob_s).mean()
        policy_loss = ((loss.detach() - self.mean_baseline['loss']) * log_prob).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy
        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss', loss)
            self.update_baseline('length', length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]
