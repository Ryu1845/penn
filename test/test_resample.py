import math

import torch

import penn


###############################################################################
# Test resample
###############################################################################


def test_resample_sine():
    """Resampling preserves frequency and has torchaudio's output length"""
    sample_rate, frequency, samples = 48000, 500., 48000
    time = torch.arange(samples) / sample_rate
    audio = torch.sin(2 * math.pi * frequency * time)[None]
    resampled = penn.resample(audio, sample_rate)

    # Length
    assert resampled.shape == (1, math.ceil(samples * penn.SAMPLE_RATE / sample_rate))

    # Spectral peak
    spectrum = torch.fft.rfft(resampled[0]).abs()
    peak = spectrum.argmax() * penn.SAMPLE_RATE / resampled.shape[-1]
    assert abs(peak - frequency) < 1.

    # Waveform away from edges
    expected = torch.sin(
        2 * math.pi * frequency *
        torch.arange(resampled.shape[-1]) / penn.SAMPLE_RATE)
    assert torch.allclose(resampled[0, 100:-100], expected[100:-100], atol=1e-3)


def test_resample_kernel_cache():
    """Resampling kernels are computed once per rate pair and device"""
    audio = torch.randn(2, 4410)
    penn.resample(audio, 44100)
    hits = penn.core._resample_kernel.cache_info().hits
    penn.resample(audio, 44100)
    assert penn.core._resample_kernel.cache_info().hits == hits + 1


def test_resample_identity():
    """Resampling to the same rate is a no-op"""
    audio = torch.randn(1, 100)
    assert penn.resample(audio, penn.SAMPLE_RATE) is audio
