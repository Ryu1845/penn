from pathlib import Path

import pytest
import torch

import penn


###############################################################################
# Constants
###############################################################################


ASSETS = Path(__file__).parent / 'assets'

# Outputs of penn 1.0.0 (torch 2.11, torbi 1.4.1, torchaudio 2.11.1)
GOLDEN = torch.load(ASSETS / 'golden' / 'gershwin.pt')


###############################################################################
# Test regression against penn 1.0.0
###############################################################################


def assert_matches(name, pitch, periodicity, min_fraction=1.):
    """Compare outputs to golden outputs"""
    expected = GOLDEN[name]
    assert pitch.shape == expected['pitch'].shape
    assert torch.allclose(periodicity, expected['periodicity'], atol=1e-5)

    # torbi and penn.viterbi resolve float32 near-ties differently; see
    # test_viterbi.py::test_pitch_decoding_is_optimal
    cents = penn.cents(pitch, expected['pitch']).abs()
    assert (cents < 1.).float().mean() >= min_fraction


def test_argmax(audio):
    pitch, periodicity = penn.from_audio(
        audio,
        penn.SAMPLE_RATE,
        center='half-hop',
        decoder='argmax')
    assert_matches('argmax', pitch, periodicity)


@pytest.mark.parametrize('batch_size', [None, 100])
def test_viterbi(audio, batch_size):
    pitch, periodicity = penn.from_audio(
        audio,
        penn.SAMPLE_RATE,
        batch_size=batch_size,
        center='half-hop',
        decoder='viterbi')
    assert_matches('viterbi', pitch, periodicity, .97)


def test_file():
    pitch, periodicity = penn.from_file(
        ASSETS / 'gershwin.wav',
        batch_size=128)
    assert_matches('file', pitch, periodicity, .97)


def test_float_hopsize(audio):
    pitch, periodicity = penn.from_audio(
        audio,
        penn.SAMPLE_RATE,
        hopsize=.0101,
        center='half-hop')
    assert_matches('float-hop', pitch, periodicity, .97)
