import torch
import torch.nn as nn
import math
from typing import Callable

__all__ = [
  'ZhuGo',
]


def kaiming_init_sequential(sequential: nn.Sequential, nonlinearity = 'leaky_relu', a = 0.01):
  for module in sequential:
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
      nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity, a=a)
      if module.bias is not None: nn.init.zeros_(module.bias)
    if isinstance(module, nn.Sequential):
      kaiming_init_sequential(module, nonlinearity, a)

class ResidualConvBlock(nn.Module):
  '''(B, C, N, M) -> (B, C, N, M)'''
  def __init__(self, channels):
    super(ResidualConvBlock, self).__init__()
    self.model = nn.Sequential(
      nn.BatchNorm2d(channels),
      nn.LeakyReLU(),
      nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(channels),
      nn.LeakyReLU(),
      nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
    )

    kaiming_init_sequential(self.model)

  def forward(self, x):
    return self.model(x) + x

class GlobalBiasBLock(nn.Module):
  '''(B, C, N, M) -> (B, C, N, M)'''
  def __init__(self, channel):
    super(GlobalBiasBLock, self).__init__()
    self.activate = nn.Sequential(
      nn.BatchNorm2d(channel),
      nn.LeakyReLU(),
    )

    self.linear = nn.Linear(2 * channel, channel)

    nn.init.xavier_normal_(self.linear.weight)
    nn.init.zeros_(self.linear.bias)

  def forward(self, x: torch.Tensor):
    y = self.activate(x)
    plane_means = y.mean(dim = (-2, -1)).flatten(start_dim = 1)
    plane_maxes = y.amax(dim = (-2, -1)).flatten(start_dim = 1)
    y = self.linear(torch.cat((plane_means, plane_maxes), dim = 1))
    y.unsqueeze_(-1).unsqueeze_(-1)
    return x + y

#
# nested bottleneck residual network
# <https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#nested-bottleneck-residual-nets>
#
#   |
#   |----.
#   |    |
#   |   BN
#   |  ReLU
#   | Conv1x1  C -> C/2
#   |    |
#   |    |----.
#   |    |    |
#   |    |   BN
#   |    |  ReLU
#   |    | Conv3x3  C/2 -> C/2
#   |    |   BN
#   |    |  ReLU
#   |    | Conv3x3  C/2 -> C/2
#   |    V    |
#   |   [+]<--`
#   |    |
#   |    |----.
#   |    |    |
#   |    |   BN
#   |    |  ReLU
#   |    | Conv3x3  C/2 -> C/2
#   |    |   BN
#   |    |  ReLU
#   |    | Conv3x3  C/2 -> C/2
#   |    V    |
#   |   [+]<--`
#   |    |
#   |   BN
#   |  ReLU
#   | Conv1x1  C/2 -> C
#   V    |
#  [+]<--`
#   |
#   V
#

class ZhuGoResidualConvBlock(nn.Module):
  '''(B, C, N, M) -> (B, C, N, M)'''
  def __init__(self, channels):
    super(ZhuGoResidualConvBlock, self).__init__()

    inner_channels = channels // 2

    self.encoder_conv1x1 = nn.Sequential(
      nn.BatchNorm2d(channels),
      nn.LeakyReLU(),
      nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False),
    )

    self.inner_residual_blocks = nn.Sequential(
      ResidualConvBlock(inner_channels),
      ResidualConvBlock(inner_channels),
    )

    self.decoder_conv1x1 = nn.Sequential(
      nn.BatchNorm2d(inner_channels),
      nn.LeakyReLU(),
      nn.Conv2d(inner_channels, channels, kernel_size=1, bias=False)
    )

    kaiming_init_sequential(self.encoder_conv1x1)
    kaiming_init_sequential(self.decoder_conv1x1)

  def forward(self, x):
    out = self.encoder_conv1x1(x)
    out = self.inner_residual_blocks(out)
    out = self.decoder_conv1x1(out) + x
    return out

class ZhuGoSharedResNet(nn.Module):
  '''(B, C1, N, M) -> (B, C2, N, M)'''
  def __init__(
    self,
    input_channels: int,
    residual_channels: int,
    residual_depths: int,
    bottleneck_channels: int,
  ):
    super(ZhuGoSharedResNet, self).__init__()

    assert len(residual_channels) == len(residual_depths) and len(residual_depths) > 0
    length = len(residual_channels)
    total_depth = 0
    residual_layers = []
    for idx, (channel, depth) in enumerate(zip(residual_channels, residual_depths)):
      for _ in range(depth):
        residual_layers.append(ZhuGoResidualConvBlock(channel))
        if total_depth % 3 == 2:
          residual_layers.append(GlobalBiasBLock(channel))
        total_depth += 1
      if idx < length - 1:
        next_channel = residual_channels[idx + 1]
        residual_layers += [
          nn.BatchNorm2d(channel),
          nn.LeakyReLU(),
          nn.Conv2d(channel, next_channel, kernel_size=3, padding=1, bias=False),
        ]

    first_channel = residual_channels[0]
    last_channel = residual_channels[-1]

    self.model = nn.Sequential(
      # input layers
      nn.Conv2d(input_channels, first_channel, kernel_size=5, padding=2, bias=False),

      *residual_layers,

      # bottleneck convolution
      nn.BatchNorm2d(last_channel),
      nn.LeakyReLU(),
      nn.Conv2d(last_channel, bottleneck_channels, kernel_size=1, bias=False),
    )

    kaiming_init_sequential(self.model)

  def forward(self, x):
    return self.model(x)

class ZhuGoPolicyHead(nn.Module):
  '''(B, C, N, M) -> (B, N * M + 1)'''
  def __init__(
    self,
    board_size: tuple[int, int],
    bottleneck_channels: int,
    policy_residual_depth: int
  ):
    super(ZhuGoPolicyHead, self).__init__()

    shared_resnet_layers = []
    for idx in range(policy_residual_depth):
      shared_resnet_layers.append(ZhuGoResidualConvBlock(bottleneck_channels))
      if idx % 3 == 0:
        shared_resnet_layers.append(GlobalBiasBLock(bottleneck_channels))

    self.shared = nn.Sequential(
      *shared_resnet_layers,
      nn.BatchNorm2d(bottleneck_channels),
      nn.LeakyReLU(),
    )

    self.move_model = nn.Sequential(
      nn.Conv2d(bottleneck_channels, bottleneck_channels // 2, kernel_size=1, bias=False),
      nn.BatchNorm2d(bottleneck_channels // 2),
      nn.LeakyReLU(),
      nn.Conv2d(bottleneck_channels // 2, 1, kernel_size=1),
      nn.Flatten(),
    )

    self.pass_model = nn.Sequential(
      nn.Conv2d(bottleneck_channels, 1, kernel_size=1, bias=False),
      nn.BatchNorm2d(1),
      nn.LeakyReLU(),
      nn.Flatten(),
      nn.Linear(board_size[0] * board_size[1], board_size[0] * board_size[1] // 4),
      nn.LeakyReLU(),
      nn.Linear(board_size[0] * board_size[1] // 4, 1),
    )

    kaiming_init_sequential(self.shared)
    kaiming_init_sequential(self.move_model)
    kaiming_init_sequential(self.pass_model)
    nn.init.xavier_normal_(self.move_model[-2].weight)
    nn.init.xavier_normal_(self.pass_model[-1].weight)

  def forward(self, x):
    out = self.shared(x)
    return torch.cat((self.move_model(out), self.pass_model(out)), dim = 1)

class ZhuGoValueHead(nn.Module):
  '''(B, C, N, M) -> (B, 1)'''
  def __init__(
    self,
    board_size: tuple[int, int],
    bottleneck_channels: int,
    value_residual_depth: int,
    value_middle_width: int
  ):
    super(ZhuGoValueHead, self).__init__()

    shared_resnet_layers = []
    for idx in range(value_residual_depth):
      shared_resnet_layers.append(ZhuGoResidualConvBlock(bottleneck_channels))
      if idx % 3 == 0:
        shared_resnet_layers.append(GlobalBiasBLock(bottleneck_channels))

    self.residual = nn.Sequential(
      *shared_resnet_layers,
      nn.BatchNorm2d(bottleneck_channels),
      nn.LeakyReLU(),
    )

    self.flatten1 = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten()
    )

    self.flatten2 = nn.Sequential(
      nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(bottleneck_channels),
      nn.LeakyReLU(),
      nn.Conv2d(bottleneck_channels, bottleneck_channels // 2, kernel_size=1, bias=False),
      nn.BatchNorm2d(bottleneck_channels // 2),
      nn.LeakyReLU(),
      nn.Conv2d(bottleneck_channels // 2, 1, kernel_size=1, bias=False),
      nn.BatchNorm2d(1),
      nn.LeakyReLU(),
      nn.Flatten(),
    )

    self.dense = nn.Sequential(
      nn.Linear(bottleneck_channels + ((board_size[0] + 1) // 2) * ((board_size[1] + 1) // 2), value_middle_width),
      nn.LeakyReLU(),
      nn.Dropout(0.5),
      nn.Linear(value_middle_width, 1)
    )

    kaiming_init_sequential(self.flatten2)
    kaiming_init_sequential(self.dense)
    nn.init.xavier_normal_(self.dense[-1].weight)
    nn.init.zeros_(self.dense[-1].bias)

  def forward(self, x):
    out = self.residual(x)
    out = torch.cat([self.flatten1(out), self.flatten2(out)], dim=1)
    out = self.dense(out)
    return out


class ZhuGo(nn.Module):
  '''
  (B, C, N, M) -> (B, N * M + 1), (B, 1)
  first is policy output logits, unnormalized, 361 (row-major moves) + 1 (pass turn).
  second is value output logits, inactivated.
  '''
  def __init__(
    self, *,
    board_size: tuple[int, int],
    input_channels : int,
    residual_channels: tuple[int],
    residual_depths: tuple[int],
    bottleneck_channels: int,
    policy_residual_depth: int,
    value_residual_depth: int,
    value_middle_width: int
  ):
    super(ZhuGo, self).__init__()

    self.shared = ZhuGoSharedResNet(input_channels, residual_channels, residual_depths, bottleneck_channels)

    self.policy = ZhuGoPolicyHead(board_size, bottleneck_channels, policy_residual_depth)

    self.value = ZhuGoValueHead(board_size, bottleneck_channels, value_residual_depth, value_middle_width)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shared = self.shared(x)
    policy = self.policy(shared)
    value = self.value(shared)
    return policy, value
