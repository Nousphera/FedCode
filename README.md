# FedCode: Communication-Efficient Federated Learning via Transferring Codebooks

FedCode communicates model-wide weight-clustering centers in most federated
rounds and periodically transfers assignment indices to calibrate local/global
models. Codebook rounds retain weight positions; calibration rounds reconstruct
client models for FedAvg.

[Paper](https://arxiv.org/abs/2311.09270) · [Provenance](PROVENANCE.md) · [Method and reproduction notes](docs/REPRODUCIBILITY.md)

<img src=./img/overview.png width=75%/>

## Quick start

```bash
git clone https://github.com/Nousphera/FedCode.git
cd FedCode
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --dataset synthetic --clients 2 --rounds 3 --epochs 1 \
  --batch-size 8 --clusters 4 --start-round 1 --broadcast-rate 3 \
  --request-rate 2 --output runs/smoke
python -m pytest -q
```

The maintained CPU runner uses the original method with stock Flower
serialization and bit-packed calibration indices. It records actual message
bytes, communication modes, configuration and model checkpoints. The small
synthetic fixture tests execution; it does not reproduce paper accuracy or DTR.

Use `--dataset cifar10` or `cifar100` for torchvision data, and `--model resnet20`
or `mobilenet`. The original audio path is available as `--dataset spcm
--model yamnet`; see the documented silence/balancing limitation before using it
as a paper benchmark. [Reproduction notes](docs/REPRODUCIBILITY.md) explain
settings, repairs, partial participation and remaining data/protocol limits.
[Validation](docs/VALIDATION.md) records the actual checks.

Original Ray orchestration is retained in `src/main.py`, and historical pins
in `requirements-legacy.txt`. [MIT license](LICENSE); architecture-specific
MIT/BSD notices are preserved in [NOTICE](NOTICE) and `licenses/`.

# References

If you find this work useful in your research, please consider citing our paper:

<pre>@misc{khalilian2023fedcode,
      title={FedCode: Communication-Efficient Federated Learning via Transferring Codebooks}, 
      author={Saeed Khalilian and Vasileios Tsouvalas and Tanir Ozcelebi and Nirvana Meratnia},
      year={2023},
      eprint={2311.09270},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
</pre>
<!-- nousphera-project-resources -->
## Project resources

[Project page](https://nousphera.github.io/FedCode/) · [Research overview](https://nousphera.github.io/projects/fedcode/) · [A0 poster](https://nousphera.github.io/FedCode/poster.pdf) · [Paper](https://arxiv.org/abs/2311.09270)

This repository contains the research implementation. The project page and poster summarize the published method and reported experiments; this documentation update does not claim an independent reproduction of those experiments.
