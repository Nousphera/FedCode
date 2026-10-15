# Testing

Activate the environment from the [README](../README.md#installation), then run:

```bash
python -m pip check
python -m pytest -q
```

The suite covers ResNet/YAMNet construction, codebook assignment, bit-packed calibration, integer buffers, bounded partitioning, late client initialization, image/audio preparation, and real training across communication modes.
Generated inputs keep regression tests small and deterministic. Dataset tests
exercise import, validation, content hashes, and consumption by the training
loader without downloading complete corpora.

## Training smoke check

```bash
python train.py --dataset synthetic --clients 2 --rounds 3 --epochs 1 \
  --batch-size 8 --clusters 4 --start-round 1 --broadcast-rate 3 \
  --request-rate 2 --output runs/smoke
```

Use a new output directory for repeated runs. End-to-end tests inspect saved
rounds, payload modes, and changed finite model parameters.

[GitHub Actions](https://github.com/Nousphera/FedCode/actions) runs the CPU checks
on Linux with Python 3.11. See each run for its commit, dependency installation,
and test output.
