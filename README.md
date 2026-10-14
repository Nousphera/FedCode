# FedCode

**FedCode: Communication-Efficient Federated Learning via Transferring Codebooks**
[Paper](https://arxiv.org/abs/2311.09270) · [Project page](https://nousphera.github.io/FedCode/) · [Code](https://github.com/Nousphera/FedCode) · [Source provenance](PROVENANCE.md)

## Abstract

Federated Learning (FL) is a distributed machine learning paradigm that enables learning models from decentralized local data. While FL offers appealing properties for clients' data privacy, it imposes high communication burdens for exchanging model weights between a server and the clients. Existing approaches rely on model compression techniques, such as pruning and weight clustering to tackle this. However, transmitting the entire set of weight updates at each federated round, even in a compressed format, limits the potential for a substantial reduction in communication volume. We propose FedCode where clients transmit only codebooks, i.e., the cluster centers of updated model weight values. To ensure a smooth learning curve and proper calibration of clusters between the server and the clients, FedCode periodically transfers model weights after multiple rounds of solely communicating codebooks. This results in a significant reduction in communication volume between clients and the server in both directions, without imposing significant computational overhead on the clients or leading to major performance degradation of the models. We evaluate the effectiveness of FedCode using various publicly available datasets with ResNet-20 and MobileNet backbone model architectures. Our evaluations demonstrate a 12.2-fold data transmission reduction on average while maintaining a comparable model performance with an average accuracy loss of 1.3% compared to FedAvg. Further validation of FedCode performance under non-IID data distributions showcased an average accuracy loss of 2.0% compared to FedAvg while achieving approximately a 12.7-fold data transmission reduction.

*The abstract describes the published work. The maintained release has not reproduced its benchmark tables.*

## Setup

Use Python 3.11. The commands below create a dedicated virtual environment.

```bash
mkdir -p "$HOME/research-code" "$HOME/.venvs"
cd "$HOME/research-code"
git clone https://github.com/Nousphera/FedCode.git
cd FedCode
python3.11 -m venv "$HOME/.venvs/fedcode"
source "$HOME/.venvs/fedcode/bin/activate"
python -m pip install -r requirements.txt
python -m pytest -q
```

## Datasets

```bash
python scripts/prepare_data.py --dataset cifar10 --root datasets --download
python scripts/prepare_data.py --dataset cifar10 --root datasets --validate
```

Use `cifar100` for CIFAR-100. Speech Commands v2 uses the original audio path;
[dataset setup](docs/DATA_SETUP.md) describes acquisition, validation, the
12-class mapping and the missing historical silence/balancing recipe. Prepared
NPZ imports are accepted by the actual trainer, with fixed dataset class counts.

## Run

```bash
python train.py --dataset synthetic --clients 2 --rounds 3 --epochs 1 \
  --batch-size 8 --clusters 4 --start-round 1 --broadcast-rate 3 \
  --request-rate 2 --output runs/smoke
python train.py --dataset cifar10 --data-root datasets --model resnet20 \
  --clients 10 --rounds 60 --epochs 4 --batch-size 128 --lr .001 \
  --clusters 64 --split noniid --alpha 10 --participation 1 --start-round 2 \
  --broadcast-rate 5 --request-rate 2 --seed 42 --output runs/cifar10-resnet
```

This maintained runner is sequential and uses stock Flower protobufs for
communication accounting. It is not a Flower App. The native algorithm retains
client weights by stable identity between codebook and calibration rounds.
Use `--model mobilenet` for MobileNet, or `--dataset spcm --model yamnet` for audio.
[Reproduction notes](docs/REPRODUCIBILITY.md) explain paper budgets, partial
participation, communication modes and the source's F1/F2 convention.

## Outputs and reproducibility

`results.json` records configuration, actual protobuf byte counts and each
round's communication mode; `model-state.pt` and `final-parameters.npz` store
final weights, and `training.log` contains the run log. Private client caches are
removed on completion. Checkpoints do not contain all optimizer/client state.
[Validation](docs/VALIDATION.md) documents maintained checks and unverified
paper accuracy/DTR claims.

## Citation

Use [CITATION.cff](CITATION.cff) or [citation.bib](citation.bib).

## License

[MIT](LICENSE), with architecture-specific MIT/BSD notices in [NOTICE](NOTICE) and `licenses/`.
