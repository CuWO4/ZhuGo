import torch
import torch.nn as nn

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
    residual_layers = []
    for idx, (channel, depth) in enumerate(zip(residual_channels, residual_depths)):
      residual_layers += [
        ZhuGoResidualConvBlock(channel) for _ in range(depth)
      ] + [
        nn.BatchNorm2d(channel),
        nn.LeakyReLU(),
      ]
      if idx < length - 1:
        next_channel = residual_channels[idx + 1]
        residual_layers.append(nn.Conv2d(channel, next_channel, kernel_size=3, padding=1, bias=False))

    first_channel = residual_channels[0]
    last_channel = residual_channels[-1]

    self.model = nn.Sequential(
      # input layers
      nn.Conv2d(input_channels, first_channel, kernel_size=5, padding=2, bias=False),

      *residual_layers,

      # bottleneck convolution
      nn.Conv2d(last_channel, bottleneck_channels, kernel_size=1, bias=False),
    )

    kaiming_init_sequential(self.model)

  def forward(self, x):
    return self.model(x)

class ZhuGoPolicyHead(nn.Module):
  '''(B, C, N, M) -> (B, N, M)'''
  def __init__(
    self,
    bottleneck_channels: int,
    policy_residual_depth: int
  ):
    super(ZhuGoPolicyHead, self).__init__()

    self.model = nn.Sequential(
      *[
        ZhuGoResidualConvBlock(bottleneck_channels)
        for _ in range(policy_residual_depth)
      ],
      nn.BatchNorm2d(bottleneck_channels),
      nn.LeakyReLU(),
      nn.Conv2d(bottleneck_channels, bottleneck_channels // 2, kernel_size=1, bias=False),
      nn.BatchNorm2d(bottleneck_channels // 2),
      nn.LeakyReLU(),
      nn.Conv2d(bottleneck_channels // 2, 1, kernel_size=1)
    )

    kaiming_init_sequential(self.model)
    nn.init.xavier_normal_(self.model[-1].weight)
    nn.init.zeros_(self.model[-1].bias)

  def forward(self, x):
    return self.model(x)

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

    self.residual = nn.Sequential(
      *[
        ZhuGoResidualConvBlock(bottleneck_channels)
        for _ in range(value_residual_depth)
      ],
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
  '''(B, C, N, M) -> [(B, N, M), (B, 1)]'''
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

    self.policy = ZhuGoPolicyHead(bottleneck_channels, policy_residual_depth)

    self.value = ZhuGoValueHead(board_size, bottleneck_channels, value_residual_depth, value_middle_width)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shared = self.shared(x)
    policy = self.policy(shared).squeeze(1) # remove channel dimension
    value = self.value(shared)
    return policy, value
