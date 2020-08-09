# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import random
from collections import defaultdict


import egg.core as core
from torch.distributions import Categorical

class CombineMMRnnSenderReinforce(nn.Module):
    def __init__(self, s1, s2):
        super(CombineMMRnnSenderReinforce, self).__init__()
        self.s1 = s1
        self.s2 = s2
    def forward(self,x):
        seq1, log1, entr1 = self.s1(x)
        seq2, log2, entr2 = self.s2(x)
        return torch.cat([seq1, seq2], dim=1), torch.cat([log1, log2], dim=1), torch.cat([entr1, entr2], dim=1)

class SplitWrapper(nn.Module):
    def __init__(self, s, part):
      super(SplitWrapper, self).__init__()
      self.s = s
      self.part = part
    def forward(self, x):
      out = self.s(x)
      return out[self.part]


class Receiver(nn.Module):
    def __init__(self, n_outputs, n_hidden):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, _):
        return self.fc(x)


class Sender(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)

    def forward(self, x):
        x = self.fc1(x)
        return x

class MMReceiver(nn.Module):
    def __init__(self, n_outputs, n_hidden):
        super(MMReceiver, self).__init__()
        self.fc = nn.Linear(2*n_hidden, n_outputs)

    def forward(self, x, _):
        return self.fc(x)


class MMSender(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(MMSender, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x1, x2

class NonLinearReceiver(nn.Module):
    def __init__(self, n_outputs, vocab_size, n_hidden, max_length):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.fc_1 = nn.Linear(vocab_size * max_length, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_outputs)

        self.diagonal_embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.eye_(self.diagonal_embedding.weight)

    def forward(self, x, *rest):
        with torch.no_grad():
            x = self.diagonal_embedding(x).view(x.size(0), -1)

        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.fc_2(x)

        zeros = torch.zeros(x.size(0), device=x.device)
        return x, zeros, zeros

class MMNonLinearReceiver(nn.Module):
    def __init__(self, n_outputs, vocab_size, n_hidden, max_length):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.fc_1 = nn.Linear(vocab_size * max_length * 2, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_outputs)

        self.diagonal_embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.eye_(self.diagonal_embedding.weight)

    def forward(self, x, *rest):
        with torch.no_grad():
            x = self.diagonal_embedding(x).view(x.size(0), -1)

        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.fc_2(x)

        zeros = torch.zeros(x.size(0), device=x.device)
        return x, zeros, zeros


class Freezer(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.eval()

    def train(self, mode):
        pass

    def forward(self, *input):
        with torch.no_grad():
            r = self.wrapped(*input)
        return r


class PlusOneWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *input):
        r1, r2, r3 = self.wrapped(*input)
        return r1 + 1, r2, r3
