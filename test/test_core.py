from pathlib import Path

import pytest
import torch

import penn


ASSETS = Path(__file__).parent / 'assets'


###############################################################################
# Test core.py
###############################################################################


def test_infer(audio):
    """Test that inference produces the correct shape"""
    pitch, periodicity = penn.from_audio(
        audio,
        penn.SAMPLE_RATE,
        center='half-hop')
    shape = (1, audio.shape[1] // penn.HOPSIZE)
    assert pitch.shape == periodicity.shape == shape


def test_infer_stereo(audio_stereo):
    """Test that inference on stereo audio produces the correct shape"""
    pitch, periodicity = penn.from_audio(
        audio_stereo,
        penn.SAMPLE_RATE,
        center='half-hop')
    shape = (1, audio_stereo.shape[1] // penn.HOPSIZE)
    assert pitch.shape == periodicity.shape == shape


def test_from_audios(audio):
    """Batched inference over many signals matches per-signal inference"""
    clips = [audio, audio[:, :8000], audio[:, 4000:20000], audio[:, :1500]]
    results = penn.from_audios(
        clips,
        penn.SAMPLE_RATE,
        batch_size=300,
        center='half-hop')
    for clip, (pitch, periodicity) in zip(clips, results):
        expected_pitch, expected_periodicity = penn.from_audio(
            clip,
            penn.SAMPLE_RATE,
            center='half-hop')
        assert pitch.shape == expected_pitch.shape
        assert (penn.cents(pitch, expected_pitch).abs() < 1.).float().mean() > .99
        assert torch.allclose(periodicity, expected_periodicity, atol=1e-5)


def test_from_audios_sample_rates(audio):
    """Signals may have different sample rates"""
    upsampled = penn.resample(audio, penn.SAMPLE_RATE, 16000)
    results = penn.from_audios(
        [audio, upsampled],
        [penn.SAMPLE_RATE, 16000],
        center='half-hop')
    assert results[0][0].shape == results[1][0].shape


def test_batch_size_invariance(audio):
    """Results do not depend on batch size"""
    pitch, periodicity = penn.from_audio(audio, penn.SAMPLE_RATE)
    batched_pitch, batched_periodicity = penn.from_audio(
        audio,
        penn.SAMPLE_RATE,
        batch_size=7)
    assert (penn.cents(pitch, batched_pitch).abs() < 1.).float().mean() > .99
    assert torch.allclose(periodicity, batched_periodicity, atol=1e-5)


def test_float_hopsize(audio):
    """Non-integer hopsizes produce the expected number of frames"""
    hopsize = .0101
    pitch, periodicity = penn.from_audio(
        audio,
        penn.SAMPLE_RATE,
        hopsize=hopsize,
        center='half-hop')
    frames = penn.expected_frames(
        audio.shape[-1],
        penn.SAMPLE_RATE,
        hopsize,
        'half-hop')
    assert pitch.shape == periodicity.shape == (1, frames)


def test_stereo_tensor():
    """Multichannel tensors at any sample rate are downmixed"""
    audio, sample_rate = penn.load.read(ASSETS / '500Hz_stereo.wav')
    assert audio.shape[0] == 2
    pitch, _ = penn.from_audio(audio, sample_rate, center='half-hop')
    frames = penn.expected_frames(
        audio.shape[-1],
        sample_rate,
        penn.HOPSIZE_SECONDS,
        'half-hop')
    assert pitch.shape == (1, frames)
    assert (penn.cents(pitch[0, 10:-10], torch.tensor(500.)).abs() < 50).all()


@pytest.mark.parametrize('num_workers', [0, 2])
def test_from_files_to_files(tmp_path, num_workers):
    """File-to-file inference matches file inference"""
    files = [ASSETS / 'gershwin.wav', ASSETS / 'beethoven.wav']
    prefixes = [tmp_path / file.stem for file in files]
    penn.from_files_to_files(
        files,
        prefixes,
        batch_size=512,
        num_workers=num_workers)
    for file, prefix in zip(files, prefixes):
        pitch = torch.load(f'{prefix}-pitch.pt')
        periodicity = torch.load(f'{prefix}-periodicity.pt')
        expected_pitch, expected_periodicity = penn.from_file(file)
        assert pitch.shape == expected_pitch.shape
        assert (penn.cents(pitch, expected_pitch).abs() < 1.).float().mean() > .99
        assert torch.allclose(periodicity, expected_periodicity, atol=1e-5)
