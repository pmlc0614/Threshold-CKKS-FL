"""CIFAR ResNet models.

The projection shortcuts and initialization follow the reference
pmlc0614/FL_RLWE2MLWE implementation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1 and in_planes == planes:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = F.relu(output, inplace=True)

        output = self.conv2(output)
        output = self.bn2(output)

        output = output + self.shortcut(inputs)
        output = F.relu(output, inplace=True)

        return output


class CifarResNet(nn.Module):
    """Projection-shortcut ResNet for 32x32 CIFAR images."""

    def __init__(
        self,
        *,
        blocks_per_stage: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        if blocks_per_stage <= 0:
            raise ValueError("blocks_per_stage must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(
            planes=16,
            block_count=blocks_per_stage,
            first_stride=1,
        )
        self.layer2 = self._make_layer(
            planes=32,
            block_count=blocks_per_stage,
            first_stride=2,
        )
        self.layer3 = self._make_layer(
            planes=64,
            block_count=blocks_per_stage,
            first_stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        planes: int,
        block_count: int,
        first_stride: int,
    ) -> nn.Sequential:
        strides = [first_stride] + [1] * (block_count - 1)
        layers: list[nn.Module] = []

        for stride in strides:
            layers.append(
                BasicBlock(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=stride,
                )
            )
            self.in_planes = planes * BasicBlock.expansion

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01,
                )
                nn.init.zeros_(module.bias)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = F.relu(output, inplace=True)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)

        output = self.avgpool(output)
        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)

        return output


class ResNet20(CifarResNet):
    """ResNet-20 for 32x32 CIFAR images."""

    def __init__(
        self,
        num_classes: int = 10,
    ) -> None:
        super().__init__(
            blocks_per_stage=3,
            num_classes=num_classes,
        )


class ResNet32(CifarResNet):
    """ResNet-32 for 32x32 CIFAR images."""

    def __init__(
        self,
        num_classes: int = 100,
    ) -> None:
        super().__init__(
            blocks_per_stage=5,
            num_classes=num_classes,
        )


def resnet20(
    num_classes: int = 10,
) -> ResNet20:
    return ResNet20(num_classes=num_classes)


def resnet32(
    num_classes: int = 100,
) -> ResNet32:
    return ResNet32(num_classes=num_classes)


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
