# Reproducibility and method correspondence

## Install and run

Use Python 3.11 and `python -m pip install -r requirements.txt`. The maintained
stack is PyTorch 2.6.0, torchvision 0.21.0, torchaudio 2.6.0, Flower 1.13.1,
NumPy 1.26.4 and scikit-learn 1.6.1. CPU is the default.

```bash
python train.py --dataset synthetic --clients 2 --rounds 3 --epochs 1 \
  --batch-size 8 --clusters 4 --start-round 1 --broadcast-rate 3 \
  --request-rate 2 --output runs/smoke
python -m pytest -q
```

The synthetic fixture has 24 training and eight test images, two classes and
seeded disjoint client shards. It checks training and communication paths only.
Every invocation needs a new output directory; a private temporary directory
holds local client states and is cleaned on normal completion or exception.
Existing cache directories are never removed on import. Outputs are
`results.json`, `model-state.pt` (state dictionary), `final-parameters.npz` and
`training.log`. `--init-state` loads a state dictionary with weights-only loading.

`train.py` calls original research clients and strategies sequentially on one
host and serializes all fit requests/responses with stock Flower protobufs.
No patched Flower or Ray is needed. The older `src/main.py` Ray launcher and
`requirements-legacy.txt` remain for comparison; their distributed execution
is not newly validated. CPU message counts are not network latency measurements.

## Communication and calibration

A model-wide K-means codebook is computed over floating-point state arrays,
including the source's floating BatchNorm state. Non-floating counters are
preserved separately. Most rounds transmit only centers. Clients map their
last local weights to nearest received centers. The server concatenates client
codebooks and maps its previous global weight positions to those centers;
**this operation is not FedAvg**.

Calibration transmits centers, bit-packed assignment indices, shape/dtype
metadata and integer buffers. The receiver reconstructs the clustered arrays.
On uplink calibration rounds, the server applies sample-weighted FedAvg to
these reconstructed client models. `--broadcast-rate 5` means one downlink
calibration every five rounds; `--request-rate 2` means one uplink calibration
every two rounds. These names follow the source because the early paper's
F1/F2 direction labels conflict. The original warmup boundary conditions are
retained; the report records the actual mode for each client in every round.

A client joining for the first time during a codebook-only round receives a
calibration payload so it has a local weight arrangement. This is recorded as
`cold_start` and included in communication totals. Successful clients retain
states by stable identity between rounds. The source previously failed when
that local state was absent under partial participation.

The new codec makes the paper's indices explicit rather than sending repeated
dense values on calibration rounds. `request_bytes` and `response_bytes` are
actual serialized protobuf sizes including headers and metadata. Compare
aggregate counts with a matching serialized baseline before claiming a DTR;
the synthetic counts do not reproduce a published compression ratio.

## Real data and paper settings

CIFAR-10/100 download through torchvision into `--data-root`. A paper-style
ResNet/CIFAR-10 full-participation IID configuration is:

```bash
python train.py --dataset cifar10 --model resnet20 --clients 10 --rounds 60 \
  --epochs 4 --batch-size 128 --lr .001 --clusters 64 \
  --split noniid --alpha 10 --participation 1 --start-round 2 \
  --broadcast-rate 5 --request-rate 2 --seed 42 --output runs/cifar10-resnet
```

The source's `--split iid` is random assignment; `--split noniid --alpha 10`
uses the paper's Dirichlet concentration for its IID setting. `--alpha .1`
selects the non-IID concentration. Non-IID partition construction checks
minimum size and has a bounded retry count. Exact paper partition files and
trial records are not supplied. Appendix budgets differ across models,
heterogeneity and participation; a single command is not the full experiment.
MobileNet is `--model mobilenet`; the paper uses a later codebook warmup and
longer budgets for that model.

SpeechCommands v2 uses `--dataset spcm --model yamnet`. The original audio
loader and power-mel-plus-offset preprocessing are retained; the model's output
width is fixed to the twelve label IDs, including when a class is absent from
loaded data. The original source does not supply an independently verified
silence-generation/unknown-class balancing protocol. Tensor shape, waveform
padding/trimming and an actual YAMNet optimization step are tested, but the
complete twelve-class paper benchmark has not been reproduced. No pretrained
AudioSet model is fetched by these commands.

## Repairs that can affect numerical results

- ResNet now constructs the actual ResNet-20 factory. YAMNet SAME padding uses
  spatial dimensions, and its classifier runs after global pooling.
- `NumPyClient.get_parameters` returns only arrays; fit loss is a scalar and
  all messages use the standard Flower contract.
- Nearest-center assignment is vectorized and preserves array dtypes; exact
  ties use the lower center. Integer counters are not clustered.
- Calibration uses explicit codebooks/indices, and first-time clients get the
  state required for subsequent codebook-only updates.
- Loss averages weight the number of examples in each batch. Training accuracy
  resets each epoch; evaluation reports examples rather than batch counts.
- Singleton IID shards, impossible Dirichlet settings and tiny positive
  participation are handled without scalar-index errors or infinite loops.
- Import no longer deletes an existing project `tmp/` directory; GPU discovery
  fails clearly when no device exists rather than sleeping forever.

Original source/history and legacy pins remain available for comparisons.
Numerical paper tables, all benchmark datasets, GPU/Ray execution and published
DTR/accuracy claims have not been independently reproduced. Read NOTICE for
model architecture attribution and third-party license texts.
