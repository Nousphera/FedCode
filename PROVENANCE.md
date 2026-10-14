# Provenance

Public source snapshot: https://github.com/vtsouval/FedCode

Source commit: `cef930a5eac50ac27be9c01317477d6a66f1ea99`. Original code and any upstream licensing files are retained. This copy adds a paper project page and poster; experiment reproduction has not been performed.

## Maintained execution release (2026-10-15)

The original client/strategy/clustering implementation is retained and repaired.
`train.py` executes it sequentially with stock Flower protobuf serialization;
`src/utils/codec.py` implements the paper's codebook/index calibration as an
explicit bit-packed payload. Numerical repairs and remaining limitations are
listed in `docs/REPRODUCIBILITY.md`. Original pins remain in
`requirements-legacy.txt`; no private history, trained checkpoint or paper
result was added. Existing website, figures and citation metadata are preserved.

The architecture sources correspond to these public implementations. Their
license texts and attribution have been restored under `licenses/` and NOTICE:

- ResNet: [akamaster/pytorch_resnet_cifar10 at d5489e8995e81e91ce6b1d69dcc98ad579b0b153](https://github.com/akamaster/pytorch_resnet_cifar10/tree/d5489e8995e81e91ce6b1d69dcc98ad579b0b153), BSD-2-Clause.
- MobileNet v2: [kuangliu/pytorch-cifar at 49b7aa97b0c12fe0d4054e670403a16b6b834ddd](https://github.com/kuangliu/pytorch-cifar/tree/49b7aa97b0c12fe0d4054e670403a16b6b834ddd), MIT.
- YAMNet: [w-hc/torch_audioset at daed697ddcd1d2f7304cc3c2b1b4295ac7e304e4](https://github.com/w-hc/torch_audioset/tree/daed697ddcd1d2f7304cc3c2b1b4295ac7e304e4), MIT.

These are inspected attribution references; they do not establish the exact
historical commits originally copied into FedCode. No pretrained weights from
these projects are bundled or downloaded by the synthetic checks.
