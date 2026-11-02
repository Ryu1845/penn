import abc
import math
from typing import Optional, Tuple

import numpy as np
import torch

import penn


###############################################################################
# Base pitch posteriorgram decoder
###############################################################################


class Decoder(abc.ABC):
    """Base decoder"""

    def __init__(self, local_expected_value=True):
        self.local_expected_value = local_expected_value

    def __call__(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a single sequence

        Arguments
            logits
                Pitch posteriorgram logits
                shape=(frames, penn.PITCH_BINS, 1)

        Returns
            bins
                Decoded pitch bins
                shape=(frames, 1)
            pitch
                Decoded pitch in Hz
                shape=(frames, 1)
        """
        bins, pitch = self.batch(logits[:, :, 0][None].to(torch.float32))
        return bins.T, pitch.T

    @abc.abstractmethod
    def batch(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a batch of sequences

        Arguments
            logits
                Pitch posteriorgram logits
                shape=(batch, frames, penn.PITCH_BINS)
            lengths
                Number of valid frames of each sequence; defaults to all
                shape=(batch,)

        Returns
            bins
                Decoded pitch bins
                shape=(batch, frames)
            pitch
                Decoded pitch in Hz
                shape=(batch, frames)
        """
        pass

    def pitch(self, bins: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Convert decoded bins to frequency in Hz"""
        if self.local_expected_value:

            # Decode using an assumption of normality around the path
            return local_expected_value(bins, logits)

        return penn.convert.bins_to_frequency(bins)


###############################################################################
# Derived pitch posteriorgram decoders
###############################################################################


class Argmax(Decoder):
    """Decode pitch using argmax"""

    def __init__(self, local_expected_value=penn.LOCAL_EXPECTED_VALUE):
        super().__init__(local_expected_value)

    def batch(self, logits, lengths=None):
        bins = logits.argmax(dim=-1)
        return bins, self.pitch(bins, logits)


class PYIN(Decoder):
    """Decode pitch via peak picking + Viterbi. Used by PYIN.

    Voiced and unvoiced states are interleaved (state = 2 * bin + unvoiced)
    so that the voicing-augmented transition matrix remains banded.
    """

    def __init__(self, local_expected_value=False):
        super().__init__(local_expected_value)
        self.bands = {}
        self.initials = {}

    def batch(self, logits, lengths=None):
        batch, frames, states = logits.shape
        device = logits.device

        # Interleave voiced and unvoiced probabilities
        voiced = torch.exp(logits)
        periodicity = penn.periodicity.sum(logits, dim=-1)
        unvoiced = ((1 - periodicity) / penn.PITCH_BINS)[..., None]
        observation = torch.stack(
            (voiced, unvoiced.expand_as(voiced)),
            dim=-1
        ).reshape(batch, frames, 2 * states)

        # Viterbi decoding
        indices = penn.viterbi.decode(
            observation,
            lengths,
            self.band(device),
            self.initial(device))

        # Unvoiced bins are offset by penn.PITCH_BINS
        bins = indices // 2 + penn.PITCH_BINS * (indices % 2)

        # Convert to frequency in Hz
        if self.local_expected_value:
            pitch = local_expected_value(
                bins.clamp(max=penn.PITCH_BINS - 1),
                logits)
        else:
            pitch = penn.convert.bins_to_frequency(bins)

        # Linearly interpolate unvoiced regions
        pitch[bins >= penn.PITCH_BINS] = 0
        pitch = torch.stack([
            torch.from_numpy(
                penn.data.preprocess.interpolate_unvoiced(
                    row.cpu().numpy().astype(np.float64))[0]).to(pitch.dtype)
            for row in pitch]).to(device)

        return bins, pitch

    def band(self, device):
        """Banded voicing-augmented transition matrix"""
        if device not in self.bands:
            transition = torch.kron(
                triangular_transition_matrix(),
                torch.tensor([[.99, .01], [.01, .99]]))
            self.bands[device] = penn.viterbi.Band.from_transition(
                transition).to(device)
        return self.bands[device]

    def initial(self, device):
        """Initial probabilities; all mass on unvoiced states"""
        if device not in self.initials:
            initial = torch.zeros(2 * penn.PITCH_BINS)
            initial[1::2] = 1 / penn.PITCH_BINS
            self.initials[device] = initial.to(device)
        return self.initials[device]


class Viterbi(Decoder):

    def __init__(self, local_expected_value=True):
        super().__init__(local_expected_value)
        self.bands = {}
        self.initials = {}

    def batch(self, logits, lengths=None):
        """Decode pitch using viterbi decoding (from librosa)"""
        device = logits.device

        # Viterbi decoding
        bins = penn.viterbi.decode(
            torch.log_softmax(logits, dim=-1),
            lengths,
            self.band(device),
            self.initial(device),
            log_probs=True)

        return bins, self.pitch(bins, logits)

    def band(self, device):
        """Banded triangular transition matrix"""
        if device not in self.bands:
            self.bands[device] = triangular_band().to(device)
        return self.bands[device]

    def initial(self, device):
        """Uniform log-initial probabilities"""
        if device not in self.initials:
            self.initials[device] = torch.full(
                (penn.PITCH_BINS,),
                math.log(1 / penn.PITCH_BINS),
                device=device)
        return self.initials[device]


###############################################################################
# Utilities
###############################################################################


def expected_value(logits, cents, dim=1):
    """Expected value computation from logits"""
    # Get local distributions
    if penn.LOSS == 'categorical_cross_entropy':
        distributions = torch.nn.functional.softmax(logits, dim=dim)
    elif penn.LOSS == 'binary_cross_entropy':
        distributions = torch.sigmoid(logits)
    else:
        raise ValueError(f'Loss {penn.LOSS} is not defined')

    # Pitch is expected value in cents
    pitch = (distributions * cents).sum(dim=dim, keepdims=True)

    # BCE requires normalization
    if penn.LOSS == 'binary_cross_entropy':
        pitch = pitch / distributions.sum(dim=dim, keepdims=True)

    # Convert to hz
    return penn.convert.cents_to_frequency(pitch)


def local_expected_value(
    bins,
    logits,
    window=penn.LOCAL_PITCH_WINDOW_SIZE):
    """Decode pitch using normal assumption around bin indices

    Arguments
        bins
            Pitch bin indices
            shape=(..., frames)
        logits
            Pitch posteriorgram logits
            shape=(..., frames, penn.PITCH_BINS)

    Returns
        pitch
            Pitch in Hz
            shape=(..., frames)
    """
    # Pad
    padded = torch.nn.functional.pad(
        logits,
        (window // 2, window // 2),
        value=-float('inf'))

    # Get indices
    indices = bins[..., None] + torch.arange(window, device=bins.device)

    # Get values in cents
    cents = penn.convert.bins_to_cents(torch.clip(indices - window // 2, 0))

    # Decode using local expected value
    return expected_value(
        torch.gather(padded, -1, indices),
        cents,
        dim=-1)[..., 0]


def local_expected_value_from_bins(
    bins,
    logits,
    window=penn.LOCAL_PITCH_WINDOW_SIZE):
    """Decode pitch using normal assumption around argmax from bin indices

    Arguments
        bins
            shape=(frames, 1)
        logits
            shape=(frames, penn.PITCH_BINS, 1)

    Returns
        pitch
            shape=(frames, 1)
    """
    return local_expected_value(bins[:, 0], logits[:, :, 0], window)[:, None]


def triangular_kernel():
    """Unnormalized triangular transition weights as a function of offset"""
    bins_per_octave = penn.OCTAVE / penn.CENTS_PER_BIN
    max_octaves_per_frame = \
        penn.MAX_OCTAVES_PER_SECOND * penn.HOPSIZE / penn.SAMPLE_RATE
    max_bins_per_frame = max_octaves_per_frame * bins_per_octave + 1
    width = min(math.ceil(max_bins_per_frame) - 1, penn.PITCH_BINS - 1)
    offsets = torch.arange(-width, width + 1)
    return torch.clip(max_bins_per_frame - offsets.abs(), 0)


def triangular_band():
    """Create a banded triangular distribution transition matrix"""
    return penn.viterbi.Band.from_kernel(triangular_kernel(), penn.PITCH_BINS)


def triangular_transition_matrix():
    """Create a triangular distribution transition matrix"""
    xx, yy = torch.meshgrid(
        torch.arange(penn.PITCH_BINS),
        torch.arange(penn.PITCH_BINS),
        indexing='ij')
    bins_per_octave = penn.OCTAVE / penn.CENTS_PER_BIN
    max_octaves_per_frame = \
        penn.MAX_OCTAVES_PER_SECOND * penn.HOPSIZE / penn.SAMPLE_RATE
    max_bins_per_frame = max_octaves_per_frame * bins_per_octave + 1
    transition = torch.clip(max_bins_per_frame - (xx - yy).abs(), 0)
    return transition / transition.sum(dim=1, keepdims=True)
