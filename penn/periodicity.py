import math

import torch

import penn


###############################################################################
# Methods for extracting a periodicity estimate from pitch posteriorgram logits
###############################################################################


def entropy(logits, dim=1):
    """Entropy-based periodicity"""
    distribution = torch.nn.functional.softmax(logits, dim=dim)
    return (
        1 + 1 / math.log(penn.PITCH_BINS) * \
        (distribution * torch.log(distribution + 1e-7)).sum(dim=dim))


def max(logits, dim=1):
    """Periodicity as the maximum confidence"""
    if penn.LOSS == 'binary_cross_entropy':
        return torch.sigmoid(logits).max(dim=dim).values
    elif penn.LOSS == 'categorical_cross_entropy':
        return torch.nn.functional.softmax(
            logits, dim=dim).max(dim=dim).values
    raise ValueError(f'Loss function {penn.LOSS} is not implemented')


def sum(logits, dim=1):
    """Periodicity as the sum of the distribution

    This is really just for PYIN, which performs a masking of the distribution
    probabilities so that it does not always add to one.
    """
    return torch.clip(torch.exp(logits).sum(dim=dim), 0, 1)
