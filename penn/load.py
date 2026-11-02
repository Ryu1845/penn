import json
import warnings

import penn


def audio(file):
    """Load mono audio from disk at penn.SAMPLE_RATE"""
    audio, sample_rate = read(file)

    # If audio is multichannel, convert to mono
    if audio.size(0) > 1:
        warnings.warn(
            f'Converting {audio.size(0)}-channel audio to mono: {file}')
        audio = audio.mean(dim=0, keepdim=True)

    # Maybe resample
    return penn.resample(audio, sample_rate)


def read(file):
    """Decode audio from disk at its native sample rate

    Returns
        audio
            Floating-point audio in [-1, 1]
            shape=(channels, samples)
        sample_rate
            The audio sample rate
    """
    # Deferred because torchcodec requires FFmpeg, which in-memory inference
    # does not
    from torchcodec.decoders import AudioDecoder

    samples = AudioDecoder(str(file)).get_all_samples()
    return samples.data, samples.sample_rate


def partition(dataset):
    """Load partitions for dataset"""
    with open(penn.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
