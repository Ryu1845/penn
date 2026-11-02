import collections
import concurrent.futures
import contextlib
import dataclasses
import functools
import math
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union)

import huggingface_hub
import torch
import torchutil

import penn


###############################################################################
# Pitch and periodicity estimation
###############################################################################


def from_audio(
        audio: torch.Tensor,
        sample_rate: int = penn.SAMPLE_RATE,
        hopsize: float = penn.HOPSIZE_SECONDS,
        fmin: float = penn.FMIN,
        fmax: float = penn.FMAX,
        checkpoint: Optional[Path] = None,
        batch_size: Optional[int] = None,
        center: str = 'half-window',
        decoder: str = penn.DECODER,
        interp_unvoiced_at: Optional[float] = None,
        gpu: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform pitch and periodicity estimation

    Args:
        audio: The audio to extract pitch and periodicity from
        sample_rate: The audio sample rate
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-window', 'half-hop', 'zero'].
        decoder: Posteriorgram decoder. One of ['argmax', 'pyin', 'viterbi'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        gpu: The index of the gpu to run inference on

    Returns:
        pitch: torch.tensor(
            shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
        periodicity: torch.tensor(
            shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
    """
    return from_audios(
        [audio],
        sample_rate,
        hopsize,
        fmin,
        fmax,
        checkpoint,
        batch_size,
        center,
        decoder,
        interp_unvoiced_at,
        gpu)[0]


def from_audios(
        audios: Sequence[torch.Tensor],
        sample_rate: Union[int, Sequence[int]] = penn.SAMPLE_RATE,
        hopsize: float = penn.HOPSIZE_SECONDS,
        fmin: float = penn.FMIN,
        fmax: float = penn.FMAX,
        checkpoint: Optional[Path] = None,
        batch_size: Optional[int] = None,
        center: str = 'half-window',
        decoder: str = penn.DECODER,
        interp_unvoiced_at: Optional[float] = None,
        gpu: Optional[int] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Perform batched pitch and periodicity estimation on many signals

    Frames from all signals are packed into shared fixed-size batches for the
    network, and many signals are decoded in a single batched Viterbi call.

    Args:
        audios: The audio signals to extract pitch and periodicity from
        sample_rate: The audio sample rate, or one sample rate per signal
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-window', 'half-hop', 'zero'].
        decoder: Posteriorgram decoder. One of ['argmax', 'pyin', 'viterbi'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        gpu: The index of the gpu to run inference on

    Returns:
        List of (pitch, periodicity) for each signal, where
            pitch: torch.tensor(
                shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
            periodicity: torch.tensor(
                shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
    """
    if isinstance(sample_rate, (int, float)):
        sample_rates = [sample_rate] * len(audios)
    else:
        sample_rates = list(sample_rate)
        if len(sample_rates) != len(audios):
            raise ValueError(
                f'Received {len(sample_rates)} sample rates for '
                f'{len(audios)} signals')

    # Inference
    engine = Engine(
        hopsize,
        fmin,
        fmax,
        checkpoint,
        batch_size,
        center,
        decoder,
        interp_unvoiced_at,
        gpu)
    results = [None] * len(audios)
    for group in engine(zip(range(len(audios)), audios, sample_rates)):
        for index, pitch, periodicity in group:
            results[index] = (pitch, periodicity)
    return results


def from_file(
        file: Path,
        hopsize: float = penn.HOPSIZE_SECONDS,
        fmin: float = penn.FMIN,
        fmax: float = penn.FMAX,
        checkpoint: Optional[Path] = None,
        batch_size: Optional[int] = None,
        center: str = 'half-window',
        decoder: str = penn.DECODER,
        interp_unvoiced_at: Optional[float] = None,
        gpu: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform pitch and periodicity estimation from audio on disk

    Args:
        file: The audio file
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-window', 'half-hop', 'zero'].
        decoder: Posteriorgram decoder. One of ['argmax', 'pyin', 'viterbi'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        gpu: The index of the gpu to run inference on

    Returns:
        pitch: torch.tensor(shape=(1, int(samples // hopsize)))
        periodicity: torch.tensor(shape=(1, int(samples // hopsize)))
    """
    # Load audio
    with torchutil.time.context('load'):
        audio, sample_rate = penn.load.read(file)

    # Inference
    return from_audio(
        audio,
        sample_rate,
        hopsize,
        fmin,
        fmax,
        checkpoint,
        batch_size,
        center,
        decoder,
        interp_unvoiced_at,
        gpu)


def from_file_to_file(
        file: Path,
        output_prefix: Optional[Path] = None,
        hopsize: float = penn.HOPSIZE_SECONDS,
        fmin: float = penn.FMIN,
        fmax: float = penn.FMAX,
        checkpoint: Optional[Path] = None,
        batch_size: Optional[int] = None,
        center: str = 'half-window',
        decoder: str = penn.DECODER,
        interp_unvoiced_at: Optional[float] = None,
        gpu: Optional[int] = None
) -> None:
    """Perform pitch and periodicity estimation from audio on disk and save

    Args:
        file: The audio file
        output_prefix: The file to save pitch and periodicity without extension
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-window', 'half-hop', 'zero'].
        decoder: Posteriorgram decoder. One of ['argmax', 'pyin', 'viterbi'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        gpu: The index of the gpu to run inference on
    """
    # Inference
    pitch, periodicity = from_file(
        file,
        hopsize,
        fmin,
        fmax,
        checkpoint,
        batch_size,
        center,
        decoder,
        interp_unvoiced_at,
        gpu)

    # Move to cpu
    with torchutil.time.context('copy-from'):
        pitch, periodicity = pitch.cpu(), periodicity.cpu()

    # Maybe use same filename with new extension
    if output_prefix is None:
        output_prefix = Path(file).parent / Path(file).stem

    # Save to disk
    with torchutil.time.context('save'):
        _save(output_prefix, pitch, periodicity)


def from_files_to_files(
    files: List[Path],
    output_prefixes: Optional[List[Path]] = None,
    hopsize: float = penn.HOPSIZE_SECONDS,
    fmin: float = penn.FMIN,
    fmax: float = penn.FMAX,
    checkpoint: Optional[Path] = None,
    batch_size: Optional[int] = None,
    center: str = 'half-window',
    decoder: str = penn.DECODER,
    interp_unvoiced_at: Optional[float] = None,
    num_workers: int = 0,
    gpu: Optional[int] = None
) -> None:
    """Perform pitch and periodicity estimation from files on disk and save

    Args:
        files: The audio files
        output_prefixes: Files to save pitch and periodicity without extension
        hopsize: The hopsize in seconds
        fmin: The minimum allowable frequency in Hz
        fmax: The maximum allowable frequency in Hz
        checkpoint: The checkpoint file
        batch_size: The number of frames per batch
        center: Padding options. One of ['half-window', 'half-hop', 'zero'].
        decoder: Posteriorgram decoder. One of ['argmax', 'pyin', 'viterbi'].
        interp_unvoiced_at: Specifies voicing threshold for interpolation
        num_workers: Number of CPU worker processes for audio decoding
        gpu: The index of the gpu to run inference on
    """
    files = [Path(file) for file in files]

    # Maybe use default output filenames
    if output_prefixes is None:
        output_prefixes = [file.parent / file.stem for file in files]

    # Decode audio in worker processes
    device = _device(gpu)
    loader = torch.utils.data.DataLoader(
        AudioFiles(files),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        prefetch_factor=4 if num_workers else None,
        multiprocessing_context=_worker_context() if num_workers else None)

    # Inference
    engine = Engine(
        hopsize,
        fmin,
        fmax,
        checkpoint,
        batch_size,
        center,
        decoder,
        interp_unvoiced_at,
        gpu)

    # Setup progress bar
    progress = torchutil.iterator(
        range(len(files)),
        penn.CONFIG,
        total=len(files))

    # Save in background threads
    futures = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        try:
            for group in engine(loader):

                # Copy results of the whole group in one transfer
                with torchutil.time.context('copy-from'):
                    group = _to_cpu(group)

                for index, pitch, periodicity in group:
                    futures.add(executor.submit(
                        _save,
                        output_prefixes[index],
                        pitch,
                        periodicity))

                # Bound the number of pending writes and raise write errors
                while len(futures) > 256:
                    done, futures = concurrent.futures.wait(
                        futures,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        future.result()

                progress.update(len(group))

            # Wait for all writes
            for future in concurrent.futures.as_completed(futures):
                future.result()

        finally:

            # Close progress bar
            progress.close()


###############################################################################
# Batched inference engine
###############################################################################


@dataclasses.dataclass(eq=False)
class Signal:
    """A signal framed lazily on the inference device"""

    # User-provided identifier
    key: Any

    # Padded mono audio at penn.SAMPLE_RATE
    audio: torch.Tensor

    # Number of output frames
    frames: int

    # Integer hopsize in samples; None if starts is used
    hopsize: Optional[int]

    # Frame start samples for non-integer hopsizes
    starts: Optional[torch.Tensor]

    # Number of frames that have been sent to the network
    cursor: int = 0

    # Network outputs
    logits: List[torch.Tensor] = dataclasses.field(default_factory=list)

    def slice(self, start: int, stop: int) -> torch.Tensor:
        """Frames [start, stop) as shape=(stop - start, 1, penn.WINDOW_SIZE)"""
        if self.starts is None:
            audio = self.audio[
                start * self.hopsize:
                (stop - 1) * self.hopsize + penn.WINDOW_SIZE]
            frames = audio.unfold(0, penn.WINDOW_SIZE, self.hopsize)
        else:
            indices = self.starts[start:stop, None] + torch.arange(
                penn.WINDOW_SIZE,
                device=self.starts.device)
            frames = self.audio[indices]
        return frames[:, None]


class Engine:
    """Batched pitch and periodicity estimation over a stream of signals

    Arguments are the same as penn.from_audio. Calling an engine on an
    iterable of (key, audio, sample_rate) yields lists of
    (key, pitch, periodicity) as groups of signals finish decoding.
    """

    def __init__(
        self,
        hopsize: float = penn.HOPSIZE_SECONDS,
        fmin: float = penn.FMIN,
        fmax: float = penn.FMAX,
        checkpoint: Optional[Path] = None,
        batch_size: Optional[int] = None,
        center: str = 'half-window',
        decoder: str = penn.DECODER,
        interp_unvoiced_at: Optional[float] = None,
        gpu: Optional[int] = None
    ):
        self.hopsize = hopsize
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.decoder = decoder
        self.interp_unvoiced_at = interp_unvoiced_at
        self.device = _device(gpu)
        self.batch_size = (
            penn.EVALUATION_BATCH_SIZE if batch_size is None else batch_size)

        # Load model
        with torchutil.time.context('model'):
            self.compiled = (
                penn.COMPILE and
                self.device.type == 'cuda' and
                penn.viterbi.core.triton_available())
            self.model = (
                _compiled_model(checkpoint, self.device) if self.compiled
                else _model(checkpoint, self.device))

    def __call__(
        self,
        signals: Iterable[Tuple[Any, torch.Tensor, int]]
    ) -> Iterator[List[Tuple[Any, torch.Tensor, torch.Tensor]]]:
        pending = collections.deque()
        completed = []
        pending_frames = 0

        for key, audio, sample_rate in signals:

            # Prepare signal for inference
            signal = self.prepare(key, audio, sample_rate)
            pending.append(signal)
            pending_frames += signal.frames

            # Infer full batches
            while pending_frames >= self.batch_size:
                pending_frames -= self.infer(pending, completed)

                # Decode groups of signals
                if sum(signal.frames for signal in completed) >= max(
                    self.batch_size,
                    penn.DECODER_BATCH_FRAMES
                ):
                    yield self.decode(completed)
                    completed = []

        # Infer final partial batch
        while pending:
            self.infer(pending, completed)

        # Decode remaining signals
        if completed:
            yield self.decode(completed)

    def decode(self, signals: List[Signal]):
        """Decode a group of signals"""
        results = []
        with torchutil.time.context('postprocess'), torch.inference_mode():
            for group in _length_groups(signals):

                # Pad to equal length
                logits = torch.nn.utils.rnn.pad_sequence(
                    [torch.cat(signal.logits) for signal in group],
                    batch_first=True)
                lengths = torch.tensor(
                    [signal.frames for signal in group],
                    device=logits.device)
                for signal in group:
                    signal.logits = []

                # Decode
                _, pitch, periodicity = _postprocess(
                    logits,
                    lengths,
                    self.fmin,
                    self.fmax,
                    self.decoder)

                for i, signal in enumerate(group):
                    signal_pitch = pitch[i:i + 1, :signal.frames]
                    signal_periodicity = periodicity[i:i + 1, :signal.frames]

                    # Maybe interpolate unvoiced regions
                    if self.interp_unvoiced_at is not None:
                        signal_pitch = penn.voicing.interpolate(
                            signal_pitch,
                            signal_periodicity,
                            self.interp_unvoiced_at)

                    results.append(
                        (signal.key, signal_pitch, signal_periodicity))

        return results

    def infer(self, pending: collections.deque, completed: List[Signal]):
        """Infer one batch of frames packed from pending signals"""
        # Pack frames
        segments, size = [], 0
        while pending and size < self.batch_size:
            signal = pending[0]
            count = min(signal.frames - signal.cursor, self.batch_size - size)
            segments.append((signal, signal.cursor, signal.cursor + count))
            signal.cursor += count
            size += count
            if signal.cursor == signal.frames:
                completed.append(pending.popleft())
        frames = torch.cat([
            signal.slice(start, stop) for signal, start, stop in segments])

        # Keep shapes static for compiled models
        if self.compiled and size < self.batch_size:
            frames = torch.nn.functional.pad(
                frames,
                (0, 0, 0, 0, 0, self.batch_size - size))

        # Infer
        with torchutil.time.context('infer'):
            with (
                torch.inference_mode(),
                _autocast(self.device),
                _cudnn_benchmark()
            ):
                logits = self.model(frames)[:size, :, 0]

            # If we're benchmarking, make sure inference finishes within timer
            if penn.BENCHMARK and self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

        # Distribute outputs
        offset = 0
        for signal, start, stop in segments:
            signal.logits.append(logits[offset:offset + stop - start])
            offset += stop - start

        return size

    def prepare(self, key, audio, sample_rate) -> Signal:
        """Prepare a signal for inference"""
        with torchutil.time.context('copy-to'):
            audio = audio.to(self.device, non_blocking=True)
        return _prepare(key, audio, sample_rate, self.hopsize, self.center)


class AudioFiles(torch.utils.data.Dataset):
    """Decode audio files at their native sample rates"""

    def __init__(self, files: Sequence[Path]):
        self.files = list(files)

    def __getitem__(self, index):
        audio, sample_rate = penn.load.read(self.files[index])
        return index, audio, sample_rate

    def __len__(self):
        return len(self.files)


###############################################################################
# Inference pipeline stages
###############################################################################


def infer(frames, checkpoint=None):
    """Forward pass through the model"""
    # Time model loading
    with torchutil.time.context('model'):
        network = _model(checkpoint, frames.device)

    # Time inference
    with torchutil.time.context('infer'):

        # Infer
        with torch.inference_mode(), _autocast(frames.device):
            logits = network(frames)

        # If we're benchmarking, make sure inference finishes within timer
        if penn.BENCHMARK and logits.device.type == 'cuda':
            torch.cuda.synchronize(logits.device)

        return logits


def postprocess(logits, fmin=penn.FMIN, fmax=penn.FMAX, decoder=penn.DECODER):
    """Convert model output to pitch and periodicity

    Arguments
        logits
            shape=(frames, penn.PITCH_BINS, 1)

    Returns
        bins, pitch, periodicity
            shape=(1, frames)
    """
    with torch.inference_mode():
        return _postprocess(
            logits[:, :, 0][None],
            None,
            fmin,
            fmax,
            decoder)


def preprocess(
    audio,
    sample_rate=penn.SAMPLE_RATE,
    hopsize=penn.HOPSIZE_SECONDS,
    batch_size=None,
    center='half-window'):
    """Convert audio to model input"""
    signal = _prepare(None, audio, sample_rate, hopsize, center)

    # Default to running all frames in a single batch
    batch_size = signal.frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, signal.frames, batch_size):
        yield signal.slice(i, min(i + batch_size, signal.frames)).contiguous()


def _save(prefix, pitch, periodicity):
    """Save pitch and periodicity to disk"""
    Path(prefix).parent.mkdir(exist_ok=True, parents=True)
    torch.save(pitch, f'{prefix}-pitch.pt')
    torch.save(periodicity, f'{prefix}-periodicity.pt')


###############################################################################
# Model loading
###############################################################################


@functools.cache
def _model(checkpoint=None, device='cpu'):
    """Load and cache a model for inference"""
    network = penn.Model()

    # Maybe download from HuggingFace
    if checkpoint is None:
        checkpoint = huggingface_hub.hf_hub_download(
            'maxrmorrison/fcnf0-plus-plus',
            'fcnf0++.pt')

    # Load from disk
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    network.load_state_dict(state['model'])

    # Prepare for inference
    return network.to(device).eval()


@functools.cache
def _compiled_model(checkpoint=None, device='cuda'):
    """Load, compile, and cache a model for inference"""
    return torch.compile(_model(checkpoint, device), dynamic=False)


###############################################################################
# Utilities
###############################################################################


def cents(a, b):
    """Compute pitch difference in cents"""
    return penn.OCTAVE * torch.log2(a / b)


def expected_frames(samples, sample_rate, hopsize, center):
    """Compute expected number of output frames"""
    # Calculate expected number of frames
    hopsize_resampled = penn.convert.seconds_to_samples(
        hopsize,
        sample_rate)
    if center == 'half-window':
        window_size_resampled = \
            penn.WINDOW_SIZE / penn.SAMPLE_RATE * sample_rate
        samples = samples - (window_size_resampled - hopsize_resampled)
    elif center == 'half-hop':
        samples = samples
    elif center == 'zero':
        samples = samples + hopsize_resampled
    else:
        raise ValueError(f'Unknown center sample {center}')
    return max(1, int(samples / hopsize_resampled))


@contextlib.contextmanager
def inference_context(model):
    device_type = next(model.parameters()).device.type

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.inference_mode():

        if device_type == 'cuda':
            with torch.autocast(device_type):
                yield
        else:
            yield

    # Prepare model for training
    model.train()


def interpolate(x, xp, fp):
    """1D linear interpolation for monotonically increasing sample points"""
    # Handle edge cases
    if xp.shape[-1] == 0:
        return x
    if xp.shape[-1] == 1:
        return torch.full(
            x.shape,
            fp.squeeze(),
            device=fp.device,
            dtype=fp.dtype)

    # Get slope and intercept using right-side first-differences
    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])
    b = fp[:, :-1] - (m.mul(xp[:, :-1]))

    # Get indices to sample slope and intercept
    indicies = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1
    indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)
    line_idx = torch.linspace(
        0,
        indicies.shape[0],
        1,
        device=indicies.device).to(torch.long).expand(indicies.shape)

    # Interpolate
    return m[line_idx, indicies].mul(x) + b[line_idx, indicies]


def normalize(frames):
    """Normalize audio frames to have mean zero and std dev one"""
    # Mean-center
    frames -= frames.mean(dim=2, keepdim=True)

    # Scale
    frames /= torch.max(
        torch.tensor(1e-10, device=frames.device),
        frames.std(dim=2, keepdim=True))

    return frames


def resample(audio, sample_rate, target_rate=penn.SAMPLE_RATE):
    """Perform audio resampling

    Bandlimited windowed-sinc interpolation equivalent to
    torchaudio.transforms.Resample with default arguments. Adapted from
    torchaudio (BSD 2-Clause License, Copyright (c) 2017 Facebook Inc.
    (Soumith Chintala)).
    """
    if sample_rate == target_rate:
        return audio
    gcd = math.gcd(int(sample_rate), int(target_rate))
    orig_freq, new_freq = int(sample_rate) // gcd, int(target_rate) // gcd
    kernel, width = _resample_kernel(orig_freq, new_freq, audio.device)

    # Pack batch
    shape = audio.size()
    audio = audio.reshape(-1, shape[-1])
    num_wavs, length = audio.shape

    # Resample
    audio = torch.nn.functional.pad(audio, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(
        audio[:, None],
        kernel.to(audio.dtype),
        stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = int(
        torch.ceil(torch.as_tensor(new_freq * length / orig_freq)))
    resampled = resampled[..., :target_length]

    # Unpack batch
    return resampled.view(shape[:-1] + resampled.shape[-1:])


def _autocast(device):
    """Mixed precision for CUDA inference"""
    if torch.device(device).type == 'cuda':
        return torch.autocast('cuda')
    return contextlib.nullcontext()


@contextlib.contextmanager
def _cudnn_benchmark():
    """Select the fastest cuDNN convolution algorithms for static shapes"""
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous


def _device(gpu):
    """Get the torch device for a gpu index"""
    return torch.device('cpu' if gpu is None else f'cuda:{gpu}')


@functools.cache
def _decoder(name):
    """Get and cache a decoder by name"""
    if name == 'argmax':
        return penn.decode.Argmax()
    if name == 'pyin':
        return penn.decode.PYIN()
    if name == 'viterbi':
        return penn.decode.Viterbi()
    raise ValueError(f'Decoder method {name} is not defined')


def _worker_context():
    """Start method for data workers; avoids forking a multithreaded process"""
    methods = torch.multiprocessing.get_all_start_methods()
    return 'forkserver' if 'forkserver' in methods else 'spawn'


def _length_groups(signals, max_padding=2.):
    """Group signals by length to bound padding when batch decoding"""
    signals = sorted(signals, key=lambda signal: signal.frames, reverse=True)
    groups, group, total = [], [], 0
    for signal in signals:
        if group and (len(group) + 1) * group[0].frames > \
                max_padding * (total + signal.frames):
            groups.append(group)
            group, total = [], 0
        group.append(signal)
        total += signal.frames
    if group:
        groups.append(group)
    return groups


def _postprocess(logits, lengths, fmin, fmax, decoder):
    """Decode a batch of logits

    Arguments
        logits
            shape=(batch, frames, penn.PITCH_BINS)
        lengths
            shape=(batch,)

    Returns
        bins, pitch, periodicity
            shape=(batch, frames)
    """
    # Copy so that callers' logits are not modified
    logits = logits.to(torch.float32, copy=True)

    # Convert frequency range to pitch bin range
    minidx = penn.convert.frequency_to_bins(torch.tensor(fmin))
    maxidx = penn.convert.frequency_to_bins(torch.tensor(fmax), torch.ceil)

    # Remove frequencies outside of allowable range
    logits[..., :minidx] = -float('inf')
    logits[..., maxidx:] = -float('inf')

    # Decode pitch from logits
    bins, pitch = _decoder(decoder).batch(logits, lengths)

    # Decode periodicity from logits
    if penn.PERIODICITY == 'entropy':
        periodicity = penn.periodicity.entropy(logits, dim=-1)
    elif penn.PERIODICITY == 'max':
        periodicity = penn.periodicity.max(logits, dim=-1)
    elif penn.PERIODICITY == 'sum':
        periodicity = penn.periodicity.sum(logits, dim=-1)
    else:
        raise ValueError(
            f'Periodicity method {penn.PERIODICITY} is not defined')

    return bins, pitch, periodicity


def _prepare(key, audio, sample_rate, hopsize, center):
    """Convert audio to a lazily-framed signal on the audio's device"""
    # Convert to mono
    audio = audio.to(torch.float32)
    if audio.dim() == 1:
        audio = audio[None]
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Get number of frames
    frames = expected_frames(audio.shape[-1], sample_rate, hopsize, center)

    # Maybe resample
    audio = resample(audio, sample_rate)

    # Maybe pad audio
    hopsize = penn.convert.seconds_to_samples(hopsize)
    if center in ['half-hop', 'zero']:
        if center == 'half-hop':
            padding = int((penn.WINDOW_SIZE - hopsize) / 2)
        else:
            padding = int(penn.WINDOW_SIZE / 2)
        audio = torch.nn.functional.pad(
            audio,
            (padding, padding),
            mode='reflect')
    audio = audio[0]

    # Integer hopsizes permit zero-copy framing
    if isinstance(hopsize, int) or hopsize.is_integer():
        hopsize = int(round(hopsize))
        starts = None
        samples = (frames - 1) * hopsize + penn.WINDOW_SIZE
    else:
        starts = (
            torch.arange(frames, dtype=torch.float64) * hopsize
        ).to(torch.float32).round().to(torch.long)
        samples = int(starts[-1]) + penn.WINDOW_SIZE
        starts = starts.to(audio.device)
        hopsize = None

    # Zero-pad frames that extend past the end of the audio
    if audio.shape[-1] < samples:
        audio = torch.nn.functional.pad(audio, (0, samples - audio.shape[-1]))

    return Signal(key, audio, frames, hopsize, starts)


@functools.lru_cache(maxsize=16)
def _resample_kernel(orig_freq, new_freq, device):
    """Windowed-sinc resampling kernel with torchaudio defaults

    Computed on CPU, as in torchaudio, so results do not depend on device.
    """
    lowpass_filter_width, rolloff = 6, .99
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = torch.arange(
        -width,
        width + orig_freq,
        dtype=torch.float64)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.).to(t), t.sin() / t)
    kernels *= window * scale
    return kernels.to(device=device, dtype=torch.float32), width


def _to_cpu(group):
    """Copy a group of results to CPU in one transfer"""
    if not group:
        return group
    keys, pitch, periodicity = zip(*group)
    lengths = [p.shape[-1] for p in pitch]
    pitch = torch.cat(pitch, dim=1).cpu().split(lengths, dim=1)
    periodicity = torch.cat(periodicity, dim=1).cpu().split(lengths, dim=1)
    return [
        (key, p.clone(), q.clone())
        for key, p, q in zip(keys, pitch, periodicity)]
