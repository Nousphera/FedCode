# Local validation — 2026-10-15

Validated on macOS arm64 with Python 3.11.15, PyTorch/torchaudio 2.6.0,
torchvision 0.21.0, Flower 1.13.1, NumPy 1.26.4 and scikit-learn 1.6.1.

`python -m pytest -q`: **10 passed**. The suite covers:

- Actual ResNet-20 construction and YAMNet batch-independent spatial padding.
- Nearest-center reconstruction, integer buffers and preserved array dtypes.
- Explicit bit-packed calibration and codebook-concatenation aggregation.
- Singleton partitions, bounded impossible Dirichlet settings and first-time
  participation in a codebook round.
- Three CPU rounds with two clients, real gradient updates, both communication
  modes and stock Flower request/response serialization.
- Audio waveform padding/trimming, spectrogram dimensions and an actual
  twelve-class YAMNet optimization step on generated waveforms.

Additional commands completed with finite loss and changed parameters:

```bash
python train.py --dataset synthetic --clients 3 --participation .34 \
  --rounds 3 --start-round 0 --epochs 1 --clusters 4 --batch-size 8 \
  --output runs/partial
python train.py --dataset synthetic --model mobilenet --clients 2 \
  --rounds 1 --epochs 1 --clusters 4 --batch-size 8 --output runs/mobilenet
```

The partial run selected one of three clients per round, initialized a late
participant through calibration, then used its cached arrangement in a
codebook-only round. `pip check`, source compilation and whitespace checks
also passed. The CI workflow is provided; its remote execution is not asserted
by this local record.

Generated images/waveforms validate execution and protocol behavior only.
No real-data accuracy, published communication ratio, full SpeechCommands
protocol, CUDA execution, distributed Ray execution or paper table has been
reproduced by these checks.


## Setup and native execution follow-up

The final Python 3.11 CPU suite passed **16 tests**. Added checks cover
prepared NPZ import/validation, wrong image shape/labels, no overwrite, content
tampering, safe tar extraction and data-loader consumption. Public corpora were
not downloaded in this check; generated source NPZ files were prepared with the
public CLI and consumed by actual training. No benchmark result is inferred.

The generated CIFAR fixture trained for two rounds through the maintained
codebook/calibration runner, retaining 40 training examples across clients.
A generated Speech Commands tree additionally validates WAV headers and splits.
