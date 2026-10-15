# Methods and configuration

`train.py` runs clients sequentially and serializes fit requests and responses
with stock Flower protobufs. It uses PyTorch models and requires no Ray worker
service. Follow the [README](../README.md) for installation and
[dataset preparation](DATA_SETUP.md) for inputs.

## Codebooks and calibration

A model-wide K-means codebook covers floating-point state arrays, including
floating BatchNorm state. Integer counters are preserved separately. In
codebook rounds, clients map their last local weights to the nearest received
centers. The server concatenates client codebooks and maps its previous global
weight positions to those centers. This operation preserves positional state
and differs from averaging client weight arrays.

Calibration sends centers, bit-packed assignment indices, shape/dtype metadata,
and integer buffers. The receiver reconstructs the clustered arrays. Uplink
calibration then applies sample-weighted FedAvg to the client models.
Nearest-center ties select the lower center; array dtypes are preserved.

- `--broadcast-rate 5`: downlink calibration every five rounds.
- `--request-rate 2`: uplink calibration every two rounds.
- `--start-round`: controls the transition from warmup to codebook exchange.

These names define directions explicitly because the early paper uses
inconsistent F1/F2 direction labels. Each round's actual mode is saved.
A client first selected during a codebook round receives a calibration payload
to initialize its local weight arrangement. This `cold_start` payload is counted
in communication totals. Successful clients retain state by stable identity.

## Models, data, and partitions

ResNet-20 and MobileNet operate on CIFAR-10/100. `--split iid` uses seeded random
assignment; `--split noniid --alpha 10` uses Dirichlet concentration 10, the
paper's approximately IID setting. Concentration 0.1 produces stronger class
heterogeneity. Construction checks minimum partition sizes with bounded retries.
Set the seed, participation rate, model, and training budget for each run.

```bash
python train.py --dataset cifar10 --data-root datasets --model resnet20 \
  --clients 10 --rounds 60 --epochs 4 --batch-size 128 --lr .001 --clusters 64 \
  --split noniid --alpha 10 --participation 1 --start-round 2 \
  --broadcast-rate 5 --request-rate 2 --seed 42 --output runs/cifar10-resnet
```

`--model mobilenet` selects MobileNet. The paper's MobileNet experiments use
later codebook activation and longer training budgets; select those options
explicitly when comparing model families.

Speech Commands uses `--dataset spcm --model yamnet`. The loader pads/trims
waveforms and computes power-mel features with an offset before taking logs.
YAMNet uses spatial SAME padding and classifies after global pooling. Its output
width is fixed to twelve labels. The data guide specifies how unknown and
silence labels enter this loader.

## Metrics and outputs

`request_bytes` and `response_bytes` count actual serialized protobuf messages,
including headers and metadata. Calculate transmission reduction against a
baseline with the same dataset, participation, training budget, and serializer.
These byte counts describe payload size; network timing is a separate metric.
Loss is weighted by examples, and training accuracy resets each epoch.

Every run needs a new output directory. Outputs are `results.json`,
`model-state.pt`, `final-parameters.npz`, and `training.log`. Temporary client
state is scoped to the invocation and cleaned on completion or exception.
`--init-state` uses weights-only loading to initialize a new run. The generated
fixture has 24 training and eight test images across two classes.

Model attribution and license notices are in [NOTICE](../NOTICE) and
[PROVENANCE.md](../PROVENANCE.md).
