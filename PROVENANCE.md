# Attribution

FedCode is developed by Saeed Khalilian, Vasileios Tsouvalas, Tanir Ozcelebi, and
Nirvana Meratnia. The public source is
[vtsouval/FedCode](https://github.com/vtsouval/FedCode), commit
`cef930a5eac50ac27be9c01317477d6a66f1ea99`. Original history and author notices
are retained under the [MIT license](LICENSE).

Model architectures adapt these public implementations:

- ResNet: [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10/tree/d5489e8995e81e91ce6b1d69dcc98ad579b0b153), BSD-2-Clause.
- MobileNet v2: [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar/tree/49b7aa97b0c12fe0d4054e670403a16b6b834ddd), MIT.
- YAMNet: [w-hc/torch_audioset](https://github.com/w-hc/torch_audioset/tree/daed697ddcd1d2f7304cc3c2b1b4295ac7e304e4), MIT.

The links identify inspected versions of the attributed implementations; exact
initial adaptation revisions are unspecified. Copyright notices and license
texts are retained in [NOTICE](NOTICE) and `licenses/`. External datasets and
model weights retain their own terms.

See [method settings](docs/METHODS.md) for the codebook and calibration protocol.
The [historical dependency transcript](docs/HISTORICAL_REQUIREMENTS.md) preserves
older pins; use the current README for installation.
