import dataclasses
import functools
import math
from typing import Optional, Union

import torch


###############################################################################
# Constants
###############################################################################


# Observation probability floor; matches torbi
LOG_TINY = math.log(torch.finfo(torch.float32).tiny)


###############################################################################
# Banded transition matrix
###############################################################################


@dataclasses.dataclass(frozen=True)
class Band:
    """Banded log-transition matrix in pull form

    The log-probability of moving from state i = j + k - width to state j is
    source[i] + values[k, j]. When values has one column, it is
    source[i] + values[k, 0], i.e., the transition depends only on the offset
    between states up to a per-source normalizer.

    Attributes
        values
            Log-transition weights
            shape=(2 * width + 1, states) or (2 * width + 1, 1)
        source
            Per-source log-normalizer
            shape=(states,)
        width
            Maximum distance between connected states
    """
    values: torch.Tensor
    source: torch.Tensor
    width: int

    @property
    def offsets(self) -> int:
        """Number of reachable source states per target state"""
        return self.values.shape[0]

    @property
    def separable(self) -> bool:
        """Whether transition weights only depend on the state offset"""
        return self.values.shape[1] == 1

    @property
    def states(self) -> int:
        """Number of states"""
        return self.source.shape[0]

    def to(self, device: Union[str, torch.device]) -> 'Band':
        """Move to device"""
        return Band(self.values.to(device), self.source.to(device), self.width)

    @classmethod
    def from_kernel(cls, kernel: torch.Tensor, states: int) -> 'Band':
        """Create a row-normalized band from a translation-invariant kernel

        Arguments
            kernel
                Unnormalized transition probability as a function of the
                offset between source and target state, i.e.,
                p(j | i) is proportional to kernel[i - j + width]
                shape=(2 * width + 1,)
            states
                Number of states

        Returns
            band
                The banded log-transition matrix
        """
        kernel = kernel.to(torch.float32)
        width = (kernel.shape[0] - 1) // 2

        # Row-normalize the dense matrix
        indices = torch.arange(states, device=kernel.device)
        offsets = indices[:, None] - indices[None] + width
        valid = (offsets >= 0) & (offsets < kernel.shape[0])
        dense = torch.where(
            valid,
            kernel[offsets.clamp(0, kernel.shape[0] - 1)],
            torch.zeros((), device=kernel.device))
        source = -torch.log(dense.sum(dim=1))

        return cls(torch.log(kernel)[:, None], source, width)

    @classmethod
    def from_transition(
        cls,
        transition: torch.Tensor,
        log_probs: bool = False
    ) -> 'Band':
        """Create a band from a dense transition matrix

        Arguments
            transition
                Transition matrix; transition[i, j] = p(state j | state i)
                shape=(states, states)
            log_probs
                Whether transition is in (natural) log space

        Returns
            band
                The banded log-transition matrix
        """
        transition = transition.to(torch.float32)
        if not log_probs:
            transition = torch.log(transition)
        states = transition.shape[0]
        device = transition.device

        # Get maximum distance between connected states
        source, target = torch.nonzero(
            torch.isfinite(transition),
            as_tuple=True)
        width = int((source - target).abs().max()) if source.numel() else 0

        # Gather band
        offsets = torch.arange(2 * width + 1, device=device)
        target = torch.arange(states, device=device)
        source = target[None] + offsets[:, None] - width
        valid = (source >= 0) & (source < states)
        values = torch.where(
            valid,
            transition[source.clamp(0, states - 1), target[None]],
            torch.full((), -float('inf'), device=device))

        return cls(values, torch.zeros(states, device=device), width)

    @classmethod
    def uniform(cls, states: int) -> 'Band':
        """Create a band with uniform transition probabilities"""
        return cls.from_kernel(torch.ones(2 * states - 1), states)


###############################################################################
# Viterbi decoding
###############################################################################


def decode(
    observation: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    transition: Optional[Union[torch.Tensor, Band]] = None,
    initial: Optional[torch.Tensor] = None,
    log_probs: bool = False,
    backend: Optional[str] = None
) -> torch.Tensor:
    """Decode the most likely state sequence of a categorical distribution

    Arguments
        observation
            Time-varying categorical distribution
            shape=(batch, frames, states)
        lengths
            Number of valid frames of each batch item; defaults to all
            shape=(batch,)
        transition
            Transition matrix, where transition[i, j] = p(state j | state i),
            or a Band; defaults to uniform
            shape=(states, states)
        initial
            Initial distribution; defaults to uniform
            shape=(states,)
        log_probs
            Whether observation, transition, and initial are in (natural)
            log space
        backend
            One of ['torch', 'triton']. Defaults to 'triton' on CUDA when
            available and 'torch' otherwise.

    Returns
        indices
            The decoded state indices; zero beyond each length
            shape=(batch, frames)
    """
    batch, frames, states = observation.shape
    device = observation.device

    # Default to decoding all frames
    if lengths is None:
        lengths = torch.full((batch,), frames, dtype=torch.long)
    lengths = lengths.to(device=device, dtype=torch.long)

    # Get banded log-transition matrix
    if transition is None:
        transition = Band.uniform(states)
    elif not isinstance(transition, Band):
        transition = Band.from_transition(transition, log_probs)
    transition = transition.to(device)

    # Get log-initial distribution
    if initial is None:
        initial = torch.full(
            (states,),
            math.log(1. / states + torch.finfo(torch.float32).tiny),
            device=device)
    else:
        initial = initial.to(device=device, dtype=torch.float32)
        if not log_probs:
            initial = torch.log(initial)

    # Get floored log-observations. This allocates the buffer that the
    # forward pass overwrites with accumulated log-probabilities.
    observation = observation.to(torch.float32)
    if not log_probs:
        observation = torch.log(observation)
    delta = torch.logaddexp(
        observation,
        torch.tensor(LOG_TINY, device=device)).contiguous()

    if frames == 0:
        return torch.zeros((batch, 0), dtype=torch.long, device=device)

    # Decode
    if backend is None:
        backend = 'triton' if device.type == 'cuda' and triton_available() \
            else 'torch'
    if backend == 'triton':
        from . import kernels
        return kernels.decode(delta, lengths, transition, initial)
    if backend == 'torch':
        forward(delta, transition, initial)
        return backtrace(delta, lengths, transition)
    raise ValueError(f'Viterbi backend {backend} is not defined')


###############################################################################
# PyTorch implementation
###############################################################################


def forward(delta: torch.Tensor, band: Band, initial: torch.Tensor) -> None:
    """In-place Viterbi forward pass

    Arguments
        delta
            Log-observations; overwritten with accumulated log-probabilities
            shape=(batch, frames, states)
        band
            Banded log-transition matrix
        initial
            Log-initial distribution
            shape=(states,)
    """
    weights = band.values.T
    delta[:, 0] += initial
    current = delta[:, 0]
    for t in range(1, delta.shape[1]):

        # Gather reachable source states for each target state
        padded = torch.nn.functional.pad(
            current + band.source,
            (band.width, band.width),
            value=-float('inf'))
        windows = padded.unfold(1, band.offsets, 1)

        # Maximize over source states
        current = (windows + weights).amax(dim=2) + delta[:, t]
        delta[:, t] = current


def backtrace(
    delta: torch.Tensor,
    lengths: torch.Tensor,
    band: Band
) -> torch.Tensor:
    """Viterbi backtrace that recomputes backpointers from the forward pass

    Arguments
        delta
            Accumulated log-probabilities from the forward pass
            shape=(batch, frames, states)
        lengths
            Number of valid frames of each batch item
            shape=(batch,)
        band
            Banded log-transition matrix

    Returns
        indices
            The decoded state indices; zero beyond each length
            shape=(batch, frames)
    """
    batch, frames, states = delta.shape
    device = delta.device
    last = lengths - 1
    weights = band.values.T
    offsets = torch.arange(band.offsets, device=device)

    # Most likely final state
    final = delta[
        torch.arange(batch, device=device),
        last.clamp(min=0)
    ].argmax(dim=1)

    indices = torch.zeros((batch, frames), dtype=torch.long, device=device)
    state = final
    for t in range(frames - 1, -1, -1):
        state = torch.where(last == t, final, state)
        active = t <= last
        indices[:, t] = torch.where(active, state, 0)
        if t == 0:
            break

        # Recompute most likely source state
        source = state[:, None] + offsets - band.width
        valid = (source >= 0) & (source < states)
        source = source.clamp(0, states - 1)
        previous = delta[:, t - 1].gather(1, source) + band.source[source]
        weight = weights if band.separable else weights[state]
        scores = torch.where(valid, previous + weight, -float('inf'))
        step = scores.argmax(dim=1)
        state = torch.where(active, state + step - band.width, state)

    return indices


###############################################################################
# Utilities
###############################################################################


@functools.cache
def triton_available() -> bool:
    """Whether Triton kernels can be used"""
    try:
        import triton
        return True
    except ImportError:
        return False
