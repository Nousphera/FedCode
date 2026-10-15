"""FedCode's ResNet20, MobileNetV2 and YAMNet backbones.

CIFAR inputs are [batch, 3, 32, 32]; speech features are [batch, 1, 64, 101].
All models train from scratch and return logits for cross-entropy training.
Component copyright and license notices are in LICENSE.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _OptionAShortcut(nn.Module):
    """CIFAR ResNet option A: subsample and zero-pad the channel dimension."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.channels // 4,
                                      self.channels // 4))


class _ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = (
            _OptionAShortcut(channels)
            if stride != 1 or in_channels != channels else nn.Sequential()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    """CIFAR ResNet-20: 3 residual blocks at each of 16, 32, and 64 channels."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._stage(16, 16, 1)
        self.layer2 = self._stage(16, 32, 2)
        self.layer3 = self._stage(32, 64, 2)
        self.linear = nn.Linear(64, num_classes)

    @staticmethod
    def _stage(in_channels: int, channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            _ResNetBlock(in_channels, channels, stride),
            _ResNetBlock(channels, channels, 1),
            _ResNetBlock(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = F.avg_pool2d(x, x.shape[3])
        return self.linear(x.flatten(1))


class _MobileNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 expansion: int, stride: int):
        super().__init__()
        self.stride = stride
        channels = expansion * in_channels
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1,
                               groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out + self.shortcut(x) if self.stride == 1 else out


class MobileNetV2(nn.Module):
    """Released CIFAR variant with ReLU, projection shortcuts, and stride-1 stem."""

    # expansion, output channels, number of blocks, first-block stride
    cfg = ((1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2),
           (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2),
           (6, 320, 1, 1))

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        layers = []
        in_channels = 32
        for expansion, out_channels, count, stride in self.cfg:
            for block_stride in [stride] + [1] * (count - 1):
                layers.append(_MobileNetBlock(in_channels, out_channels,
                                              expansion, block_stride))
                in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, 4)
        return self.linear(x.flatten(1))


class _SameConv2d(nn.Conv2d):
    """TensorFlow SAME padding, including asymmetric padding at stride > 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = []
        for size, kernel, stride, dilation in zip(
                x.shape[-2:], self.kernel_size, self.stride, self.dilation):
            output_size = (size + stride - 1) // stride
            effective_kernel = (kernel - 1) * dilation + 1
            padding.append(max(0, (output_size - 1) * stride
                               + effective_kernel - size))
        height, width = padding
        if height % 2 or width % 2:
            x = F.pad(x, (0, width % 2, 0, height % 2))
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        (height // 2, width // 2), self.dilation, self.groups)


class _YAMNetConvBNReLU(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(conv.out_channels, eps=1e-4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _YAMNetConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.fused = _YAMNetConvBNReLU(
            _SameConv2d(in_channels, out_channels, 3, stride, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fused(x)


class _YAMNetSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.depthwise_conv = _YAMNetConvBNReLU(
            _SameConv2d(in_channels, in_channels, 3, stride,
                        groups=in_channels, bias=False))
        self.pointwise_conv = _YAMNetConvBNReLU(
            _SameConv2d(in_channels, out_channels, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class YAMNet(nn.Module):
    """Full 14-stage YAMNet with a randomly initialized 12-class head."""

    # output channels, stride; stage 1 is dense, stages 2-14 are separable.
    cfg = ((32, 2), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
           (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
           (1024, 2), (1024, 1))

    def __init__(self, num_classes: int = 12):
        super().__init__()
        in_channels = 1
        for index, (out_channels, stride) in enumerate(self.cfg, start=1):
            layer = _YAMNetConv if index == 1 else _YAMNetSeparableConv
            self.add_module(f"layer{index}",
                            layer(in_channels, out_channels, stride))
            in_channels = out_channels
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index in range(1, len(self.cfg) + 1):
            x = getattr(self, f"layer{index}")(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.classifier(x.flatten(1))


def make_model(name: str, num_classes: int) -> nn.Module:
    """Build a full paper backbone without pretrained weights or network access."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    constructors = {"resnet20": ResNet20, "mobilenetv2": MobileNetV2,
                    "yamnet": YAMNet}
    try:
        constructor = constructors[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; choose {', '.join(constructors)}") from exc
    return constructor(num_classes=num_classes)
