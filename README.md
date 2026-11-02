<h1 align="center">Pitch-Estimating Neural Networks (PENN)</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/penn.svg)](https://pypi.python.org/pypi/penn)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/penn)](https://pepy.tech/project/penn)

</div>

Training, evaluation, and inference of neural pitch and periodicity estimators in PyTorch. Includes the original code for the paper ["Cross-domain Neural Pitch and Periodicity Estimation"](https://arxiv.org/abs/2301.12258).


## Table of contents

- [Installation](#installation)
- [Inference](#inference)
    * [Application programming interface](#application-programming-interface)
        * [`penn.from_audio`](#pennfrom_audio)
        * [`penn.from_audios`](#pennfrom_audios)
        * [`penn.from_file`](#pennfrom_file)
        * [`penn.from_file_to_file`](#pennfrom_file_to_file)
        * [`penn.from_files_to_files`](#pennfrom_files_to_files)
    * [Command-line interface](#command-line-interface)
- [Training](#training)
    * [Download](#download)
    * [Preprocess](#preprocess)
    * [Partition](#partition)
    * [Train](#train)
    * [Monitor](#monitor)
- [Evaluation](#evaluation)
    * [Evaluate](#evaluate)
    * [Plot](#plot)
- [Citation](#citation)


## Installation

If you want to perform pitch estimation using a pretrained FCNF0++ model, run
`pip install penn`

If you want to train or use your own models, run
`pip install penn[train]`

Loading audio from disk uses [TorchCodec](https://github.com/pytorch/torchcodec), which requires FFmpeg (e.g., `sudo apt install ffmpeg` or `conda install ffmpeg`). Inference on audio tensors does not require FFmpeg. Python 3.12 or later and PyTorch 2.15 or later are required.


## Inference

Perform inference using FCNF0++

```
import penn

# Load audio
audio, sample_rate = penn.load.read('test/assets/gershwin.wav')

# Here we'll use a 10 millisecond hopsize
hopsize = .01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# Choose a gpu index to use for inference. Set to None to use cpu.
gpu = 0

# If you are using a gpu, pick a batch size that doesn't cause memory errors
# on your gpu
batch_size = 2048

# Select a checkpoint to use for inference. Selecting None will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = 'half-hop'

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = .065

# (Optional) Select a decoding method. One of ['argmax', 'pyin', 'viterbi'].
decoder = 'viterbi'

# Infer pitch and periodicity
pitch, periodicity = penn.from_audio(
    audio,
    sample_rate,
    hopsize=hopsize,
    fmin=fmin,
    fmax=fmax,
    checkpoint=checkpoint,
    batch_size=batch_size,
    center=center,
    decoder=decoder,
    interp_unvoiced_at=interp_unvoiced_at,
    gpu=gpu)
```

Note that pitch estimation is performed independently on each frame of audio. Then, a _decoding_ step occurs, which may or may not be computed independently on each frame. Most often, Viterbi decoding is used (as in, e.g., PYIN and CREPE). `penn` includes a banded Viterbi decoder (`penn.viterbi.decode`) that exploits the bounded pitch velocity of the transition matrix. On CUDA, it runs as a Triton kernel that decodes each sequence in a single launch; otherwise it runs in PyTorch.

On CUDA, the network is compiled with `torch.compile` by default. The first call compiles the network, which takes tens of seconds. Disable compilation by setting `COMPILE = False` in a configuration file. For the highest throughput on many signals, use `penn.from_audios` or `penn.from_files_to_files`. These pack frames of many signals into full batches and decode many signals in one Viterbi call.


### Application programming interface

#### `penn.from_audio`

```
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
    interp_unvoiced_at: Specifies voicing threshold for interpolation
    gpu: The index of the gpu to run inference on

Returns:
    pitch: torch.tensor(
        shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
    periodicity: torch.tensor(
        shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
"""
```


#### `penn.from_audios`

```
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
```


#### `penn.from_file`

```
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
    interp_unvoiced_at: Specifies voicing threshold for interpolation
    gpu: The index of the gpu to run inference on

Returns:
    pitch: torch.tensor(shape=(1, int(samples // hopsize)))
    periodicity: torch.tensor(shape=(1, int(samples // hopsize)))
"""
```


#### `penn.from_file_to_file`

```
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
    interp_unvoiced_at: Specifies voicing threshold for interpolation
    gpu: The index of the gpu to run inference on
"""
```


#### `penn.from_files_to_files`

```
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
    interp_unvoiced_at: Specifies voicing threshold for interpolation
    num_workers: Number of CPU worker processes for audio decoding
    gpu: The index of the gpu to run inference on
"""
```


### Command-line interface

```
python -m penn
    --files FILES [FILES ...]
    [-h]
    [--config CONFIG]
    [--output_prefixes OUTPUT_PREFIXES [OUTPUT_PREFIXES ...]]
    [--hopsize HOPSIZE]
    [--fmin FMIN]
    [--fmax FMAX]
    [--checkpoint CHECKPOINT]
    [--batch_size BATCH_SIZE]
    [--center {half-window,half-hop,zero}]
    [--decoder {argmax,pyin,viterbi}]
    [--interp_unvoiced_at INTERP_UNVOICED_AT]
    [--num_workers NUM_WORKERS]
    [--gpu GPU]

required arguments:
    --files FILES [FILES ...]
        The audio files to process

optional arguments:
    -h, --help
        show this help message and exit
    --config CONFIG
        The configuration file. Defaults to using FCNF0++.
    --output_prefixes OUTPUT_PREFIXES [OUTPUT_PREFIXES ...]
        The files to save pitch and periodicity without extension.
        Defaults to files without extensions.
    --hopsize HOPSIZE
        The hopsize in seconds. Defaults to 0.01 seconds.
    --fmin FMIN
        The minimum frequency allowed in Hz. Defaults to 31.0 Hz.
    --fmax FMAX
        The maximum frequency allowed in Hz. Defaults to 1984.0 Hz.
    --checkpoint CHECKPOINT
        The model checkpoint file. Defaults to ./penn/assets/checkpoints/fcnf0++.pt.
    --batch_size BATCH_SIZE
        The number of frames per batch. Defaults to 2048.
    --center {half-window,half-hop,zero}
        Padding options
    --decoder {argmax,pyin,viterbi}
        Posteriorgram decoder
    --interp_unvoiced_at INTERP_UNVOICED_AT
        Specifies voicing threshold for interpolation. Defaults to 0.1625.
    --num_workers NUM_WORKERS
        Number of CPU worker processes for audio decoding
    --gpu GPU
        The index of the gpu to perform inference on. Defaults to CPU.
```


## Training

### Download

`python -m penn.data.download`

Downloads and uncompresses the `mdb` and `ptdb` datasets used for training.


### Preprocess

`python -m penn.data.preprocess --config <config>`

Converts each dataset to a common format on disk ready for training. You
can optionally pass a configuration file to override the default configuration.


### Partition

`python -m penn.partition`

Generates `train`, `valid`, and `test` partitions for `mdb` and `ptdb`.
Partitioning is deterministic given the same random seed. You do not need to
run this step, as the original partitions are saved in
`penn/assets/partitions`.


### Train

`python -m penn.train --config <config> --gpu <gpu>`

Trains a model according to a given configuration on the `mdb` and `ptdb`
datasets.


### Monitor

You can monitor training via `tensorboard`.

```
tensorboard --logdir runs/ --port <port> --load_fast true
```

To use the `torchutil` notification system to receive notifications for long
jobs (download, preprocess, train, and evaluate), set the
`PYTORCH_NOTIFICATION_URL` environment variable to a supported webhook as
explained in [the Apprise documentation](https://pypi.org/project/apprise/).


## Evaluation

### Evaluate

```
python -m penn.evaluate \
    --config <config> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Evaluate a model. `<checkpoint>` is the checkpoint file to evaluate and `<gpu>`
is the GPU index.


### Plot

```
python -m penn.plot.density \
    --config <config> \
    --true_datasets <true_datasets> \
    --inference_datasets <inference_datasets> \
    --output_file <output_file> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Plot the data distribution and inferred distribution for a given dataset and
save to a jpg file.

```
python -m penn.plot.logits \
    --config <config> \
    --audio_file <audio_file> \
    --output_file <output_file> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Plot the pitch posteriorgram of an audio file and save to a jpg file.

```
python -m penn.plot.threshold \
    --names <names> \
    --evaluations <evaluations> \
    --output_file <output_file>
```

Plot the periodicity performance (voiced/unvoiced F1) over mdb and ptdb as a
function of the voiced/unvoiced threshold. `names` are the plot labels to give
each evaluation. `evaluations` are the names of the evaluations to plot.


## Citation

### IEEE
M. Morrison, C. Hsieh, N. Pruyne, and B. Pardo, "Cross-domain Neural Pitch and Periodicity Estimation," arXiv preprint arXiv:2301.12258, 2023.


### BibTex

```
@inproceedings{morrison2023cross,
    title={Cross-domain Neural Pitch and Periodicity Estimation},
    author={Morrison, Max and Hsieh, Caedon and Pruyne, Nathan and Pardo, Bryan},
    booktitle={arXiv preprint arXiv:2301.12258},
    year={2023}
}
