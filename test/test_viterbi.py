import os

import numpy as np
import pytest
import torch

import penn


###############################################################################
# Reference implementation
###############################################################################


def dense_viterbi(observation, transition, initial):
    """Float64 Viterbi decoding with a dense transition matrix"""
    with np.errstate(divide='ignore'):
        observation = np.logaddexp(
            np.log(observation.astype(np.float64)),
            penn.viterbi.core.LOG_TINY)
        transition = np.log(transition.astype(np.float64))
        delta = np.log(initial.astype(np.float64)) + observation[0]
    pointers = np.zeros(observation.shape, dtype=int)
    for t in range(1, len(observation)):
        scores = delta[:, None] + transition
        pointers[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + observation[t]
    path = [delta.argmax()]
    for t in range(len(observation) - 1, 0, -1):
        path.append(pointers[t, path[-1]])
    return np.array(path[::-1])


def path_log_prob(path, observation, transition, initial):
    """Float64 log-probability of a state sequence"""
    with np.errstate(divide='ignore'):
        observation = np.logaddexp(
            np.log(observation.astype(np.float64)),
            penn.viterbi.core.LOG_TINY)
        transition = np.log(transition.astype(np.float64))
        score = np.log(initial[path[0]]) + observation[0, path[0]]
    steps = np.arange(1, len(path))
    return score + (
        transition[path[:-1], path[1:]] + observation[steps, path[1:]]).sum()


def random_problem(states, frames, width, batch=1, seed=0):
    """Random banded decoding problem"""
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.arange(states)[:, None] - torch.arange(states)[None]
    transition = torch.rand(states, states, generator=generator)
    transition = transition * (offsets.abs() <= width)
    transition /= transition.sum(dim=1, keepdim=True)
    observation = torch.softmax(
        3 * torch.randn(batch, frames, states, generator=generator),
        dim=-1)
    initial = torch.softmax(torch.randn(states, generator=generator), dim=0)
    return observation, transition, initial


###############################################################################
# Test viterbi
###############################################################################


@pytest.mark.parametrize('width', [0, 4, 63])
def test_banded_matches_dense(width):
    """Banded decoding matches dense reference decoding"""
    for seed in range(5):
        observation, transition, initial = random_problem(64, 40, width, seed=seed)
        indices = penn.viterbi.decode(
            observation,
            transition=transition,
            initial=initial,
            backend='torch')
        expected = dense_viterbi(
            observation[0].numpy(),
            transition.numpy(),
            initial.numpy())
        assert (indices[0].numpy() == expected).all()


def test_lengths():
    """Batched decoding with variable lengths matches individual decoding"""
    observation, transition, initial = random_problem(48, 30, 5, batch=3)
    lengths = torch.tensor([30, 11, 1])
    indices = penn.viterbi.decode(
        observation,
        lengths,
        transition,
        initial,
        backend='torch')
    for i, length in enumerate(lengths):
        expected = penn.viterbi.decode(
            observation[i:i + 1, :length],
            transition=transition,
            initial=initial,
            backend='torch')
        assert torch.equal(indices[i, :length], expected[0])
        assert (indices[i, length:] == 0).all()


def test_kernel_band_matches_dense():
    """Translation-invariant bands match their dense transition matrix"""
    kernel = torch.tensor([1., 2., 4., 2., 1.])
    states = 32
    offsets = torch.arange(states)[:, None] - torch.arange(states)[None]
    dense = torch.where(
        offsets.abs() <= 2,
        kernel[(offsets + 2).clamp(0, 4)],
        torch.zeros(()))
    dense /= dense.sum(dim=1, keepdim=True)
    observation = torch.softmax(
        2 * torch.randn(1, 50, states, generator=torch.Generator().manual_seed(0)),
        dim=-1)
    indices = penn.viterbi.decode(
        observation,
        transition=penn.viterbi.Band.from_kernel(kernel, states))
    expected = dense_viterbi(
        observation[0].numpy(),
        dense.numpy(),
        np.full(states, 1 / states))
    assert (indices[0].numpy() == expected).all()


def test_pitch_decoding_is_optimal(audio):
    """Viterbi pitch decoding finds the float64-optimal path"""
    logits = torch.cat([
        penn.infer(frames) for frames in
        penn.preprocess(audio, penn.SAMPLE_RATE, center='half-hop')])
    bins, _, _ = penn.postprocess(logits, decoder='viterbi')

    # Exact decoding with the dense transition matrix
    minidx = penn.convert.frequency_to_bins(torch.tensor(penn.FMIN))
    maxidx = penn.convert.frequency_to_bins(torch.tensor(penn.FMAX), torch.ceil)
    logits = logits[:, :, 0].clone()
    logits[:, :minidx] = -float('inf')
    logits[:, maxidx:] = -float('inf')
    observation = torch.softmax(logits, dim=-1).numpy()
    transition = penn.decode.triangular_transition_matrix().numpy()
    initial = np.full(penn.PITCH_BINS, 1 / penn.PITCH_BINS)
    expected = dense_viterbi(observation, transition, initial)

    # Near-ties at float32 precision may select a different path
    path = bins[0].numpy()
    assert (path == expected).mean() > .99
    assert path_log_prob(path, observation, transition, initial) > \
        path_log_prob(expected, observation, transition, initial) - 1e-3


def test_pyin_interleaved_matches_dense():
    """Interleaved PYIN decoding matches the original state ordering"""
    generator = torch.Generator().manual_seed(0)
    logits = torch.log_softmax(
        4 * torch.randn(1, 12, penn.PITCH_BINS, generator=generator),
        dim=-1)
    logits[..., ::3] -= 2.
    bins, _ = penn.decode.PYIN().batch(logits)

    # Original layout: voiced states, then unvoiced states
    periodicity = penn.periodicity.sum(logits, dim=-1)[0]
    unvoiced = ((1 - periodicity) / penn.PITCH_BINS)[:, None]
    observation = torch.cat(
        (torch.exp(logits[0]), unvoiced.expand(-1, penn.PITCH_BINS)),
        dim=1)
    transition = torch.kron(
        torch.tensor([[.99, .01], [.01, .99]]),
        penn.decode.triangular_transition_matrix())
    initial = torch.zeros(2 * penn.PITCH_BINS)
    initial[penn.PITCH_BINS:] = 1 / penn.PITCH_BINS
    expected = dense_viterbi(
        observation.numpy(),
        transition.numpy(),
        initial.numpy())
    assert (bins[0].numpy() == expected).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1',
    reason='Requires CUDA or TRITON_INTERPRET=1')
def test_triton_matches_torch():
    """Triton kernels exactly match the PyTorch implementation"""
    pytest.importorskip('triton')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generic band
    observation, transition, initial = random_problem(100, 20, 9, batch=3)
    lengths = torch.tensor([20, 13, 1])
    expected = penn.viterbi.decode(
        observation,
        lengths,
        transition,
        initial,
        backend='torch')
    indices = penn.viterbi.decode(
        observation.to(device),
        lengths,
        transition,
        initial,
        backend='triton')
    assert torch.equal(indices.cpu(), expected)

    # Translation-invariant pitch band
    generator = torch.Generator().manual_seed(1)
    observation = torch.log_softmax(
        4 * torch.randn(2, 12, penn.PITCH_BINS, generator=generator),
        dim=-1)
    lengths = torch.tensor([12, 7])
    band = penn.decode.triangular_band()
    expected = penn.viterbi.decode(
        observation,
        lengths,
        band,
        log_probs=True,
        backend='torch')
    indices = penn.viterbi.decode(
        observation.to(device),
        lengths,
        band,
        log_probs=True,
        backend='triton')
    assert torch.equal(indices.cpu(), expected)
