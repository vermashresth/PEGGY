# PEGGY 🐣: Population based Emergence of lanGuage in Games

![GitHub](https://img.shields.io/github/license/facebookresearch/EGG)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Introduction

EGG (Emergence of Language in Games) is a toolkit that allows researchers to quickly implement multi-agent games with discrete channel communication. In 
such games, the agents are trained to communicate with each other and jointly solve a task. Often, the way they communicate is not explicitly determined, allowing agents to come up with their own 'language' in order to solve the task.
Such setup opens a plethora of possibilities in studying emergent language and the impact of the nature of task being solved, the agents' models, etc. This subject is a vibrant area of research often considered as a prerequisite for general AI. The purpose of EGG is to offer researchers an easy and fast entry point into this research area.

PEGGY builds up on EGG by including population based experiments in Language Games. Further, advising mechanism are also implemented for improving sample efficiency.


The codebase is in [PyTorch](https://pytorch.org/) and provides: (a) simple, yet powerful components for implementing 
communication between agents, (b) a diverse set of pre-implemented games, (c) an interface to analyse the emergent 
communication protocols.

Key features:
 * Primitives for implementing a single-symbol or variable-length communication (with vanilla RNNs, GRUs, LSTMs or Transformers);
 * Training with optimization of the communication channel with REINFORCE or Gumbel-Softmax relaxation via a common interface;
 * Simplified configuration of the general components, such as checkpointing, optimization, tensorboard support, etc;
 * Provides a simple CUDA-aware command-line tool for grid-search over parameters of the games.
 * Simple wrappers to extend two agent games to populations

Please look at EGG's documentation for full list of implemented games and documentation

## Testing
Run pytest:

```
python -m pytest
```

All tests should pass.

