# Dataset preparation

Run commands from the repository root in the README's environment.
`python scripts/prepare_data.py --help` lists datasets and modes.
`--download` explicitly enables network acquisition; `--source` imports a safe
NPZ or already-prepared directory. Import rejects existing destination files and
symlinks in directory trees. Imports validate in a temporary staging directory before publishing; failed imports leave no dataset or manifest behind. Directory imports copy only the selected dataset subtree, and auxiliary imports copy only supported media files. `--validate` reads schema and content again and
compares the deterministic SHA-256 manifest; it never downloads. Omitting all
three modes registers an already-present dataset after validating its schema.

Prepared image files contain `train_images`, `train_labels`, `test_images`, and
`test_labels`. RGB images are uint8 NHWC with shape `(N,32,32,3)` for CIFAR.
Labels are integer IDs shaped `(N,)` or `(N,1)`. Preparation checks shapes,
class ranges, nonempty splits, and counts. Training reads these files directly
and preserves their row order.

```bash
python scripts/prepare_data.py --dataset cifar10 --root datasets --source /path/to/cifar10.npz
python scripts/prepare_data.py --dataset cifar10 --root datasets --validate
```

A manifest records dataset name, split sizes, class count and relative file
paths/content hashes. Retain the source publication, data license, and acquisition record alongside
this manifest. A changed file requires a new destination/preparation run;
manifests are not silently regenerated over changed data.

## Public vision sources

[CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) use the official
train/test splits via the pinned torchvision loader. Files are saved
as `ROOT/cifar10.npz` and `ROOT/cifar100.npz` for training:

```bash
python scripts/prepare_data.py --dataset cifar10 --root datasets --download
python scripts/prepare_data.py --dataset cifar100 --root datasets --download
```

## Speech Commands v2

```bash
python scripts/prepare_data.py --dataset spcm --root datasets --download
python scripts/prepare_data.py --dataset spcm --root datasets --validate
python train.py --dataset spcm --model yamnet --data-root datasets --output runs/spcm
```

[Speech Commands v2](https://www.tensorflow.org/datasets/catalog/speech_commands)
is downloaded by the pinned torchaudio dataset implementation. The extracted
layout is `ROOT/spcm/speech_commands_v0.02/`, including official testing and
validation lists. Existing prepared roots can be imported with `--source` into
a new root; no arbitrary archive is imported. Validation reads each split's
metadata, checks 16 kHz audio declarations and hashes the corpus.

The loader maps ten command names plus unknown and silence to twelve output
IDs. All other spoken words map to unknown. The default archive reader uses the
provided recordings without additional unknown-class balancing or synthetic
silence generation. A class can therefore have no examples while retaining its
output ID. Record any supplied silence samples or balancing changes explicitly.
Power-mel preprocessing and YAMNet training are specified in
[METHODS.md](METHODS.md).

Speech Commands waveform loading uses the SoundFile backend installed by `requirements.txt`. Preparation checks complete PCM payloads before the training reader decodes them.
