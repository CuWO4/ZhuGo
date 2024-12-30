import torch
import torch.nn as nn

__all__ = [
  'ZhuGo',
]

class ResidualConvBlock(nn.Module):
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

    nn.init.kaiming_normal_(self.model[2].weight)
    nn.init.kaiming_normal_(self.model[5].weight)

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
  def __init__(self, channels):
    super(ZhuGoResidualConvBlock, self).__init__()

    inner_channels = channels // 2
    
    self.encoder_conv1x1 = nn.Sequential(
      nn.BatchNorm2d(channels),
      nn.LeakyReLU(),
      nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False),
    )
    
    self.inner_residual_block1 = ResidualConvBlock(inner_channels)
    self.inner_residual_block2 = ResidualConvBlock(inner_channels)
    
    self.decoder_conv1x1 = nn.Sequential(
      nn.BatchNorm2d(inner_channels),
      nn.LeakyReLU(),
      nn.Conv2d(inner_channels, channels, kernel_size=1, bias=False)
    )

    nn.init.kaiming_normal_(self.encoder_conv1x1[-1].weight)
    nn.init.kaiming_normal_(self.decoder_conv1x1[-1].weight)

  def forward(self, x):
    out = self.encoder_conv1x1(x)
    out = self.inner_residual_block1(out) + out
    out = self.inner_residual_block2(out) + out
    out = self.decoder_conv1x1(out)
    return out

class MultiScaleConvBlock(nn.Module):
  def __init__(self, input_channels, output_channels, bias=True):
    super(MultiScaleConvBlock, self).__init__()
    channels7x7 = output_channels // 8
    channels5x5 = output_channels // 4
    channels3x3 = output_channels - channels7x7 - channels5x5

    self.conv3x3 = nn.Conv2d(input_channels, channels3x3, kernel_size=3, padding=1, bias=bias)
    self.conv5x5 = nn.Conv2d(input_channels, channels5x5, kernel_size=5, padding=2, bias=bias)
    self.conv7x7 = nn.Conv2d(input_channels, channels7x7, kernel_size=7, padding=3, bias=bias)

    nn.init.kaiming_normal_(self.conv3x3.weight)
    nn.init.kaiming_normal_(self.conv5x5.weight)
    nn.init.kaiming_normal_(self.conv7x7.weight)

  def forward(self, x):
    y3x3 = self.conv3x3(x)
    y5x5 = self.conv5x5(x)
    y7x7 = self.conv7x7(x)
    return torch.cat([y3x3, y5x5, y7x7], dim=1)

class ZhuGo(nn.Module):
  def __init__(
    self, *,
    board_size: tuple[int, int],
    input_channels : int,
    # note  each depth of residual block contain 2 convolution layers
    residual_channels: tuple[int], 
    residual_depths: tuple[int], 
    bottleneck_channels: int,
    policy_middle_channel: int,
    value_middle_width: int
  ):
    super(ZhuGo, self).__init__()

    # shared

    assert len(residual_channels) == len(residual_depths) and len(residual_depths) > 0
    length = len(residual_channels)
    residual_layers = []
    for idx, (channel, depth) in enumerate(zip(residual_channels, residual_depths)):
      residual_layers += [ZhuGoResidualConvBlock(channel) for _ in range(depth)] + [
        nn.BatchNorm2d(channel),
        nn.LeakyReLU(),
      ]
      if idx < length - 1:
        next_channel = residual_channels[idx + 1]
        residual_layers += [
          nn.Dropout2d(0.1),
          MultiScaleConvBlock(channel, next_channel, bias=False),
        ]

    first_channel = residual_channels[0]
    last_channel = residual_channels[-1]

    self.shared = nn.Sequential(
      # input layers
      nn.Conv2d(input_channels, first_channel, kernel_size=5, padding=2, bias=False),

      # residual layers
      *residual_layers,

      # Bottleneck convolution
      nn.Conv2d(last_channel, bottleneck_channels, kernel_size=1, bias=False),
      nn.BatchNorm2d(bottleneck_channels),
      nn.LeakyReLU(),
    )
    
    nn.init.kaiming_normal_(self.shared[0].weight)
    nn.init.kaiming_normal_(self.shared[-3].weight)

    # policy head

    self.policy = nn.Sequential(
      nn.Conv2d(bottleneck_channels, policy_middle_channel, kernel_size=5, padding=2, bias=False),
      nn.BatchNorm2d(policy_middle_channel),
      nn.LeakyReLU(),
      nn.Conv2d(policy_middle_channel, policy_middle_channel, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(policy_middle_channel),
      nn.LeakyReLU(),
      nn.Conv2d(policy_middle_channel, 1, kernel_size=1)
    )
    
    nn.init.kaiming_normal_(self.policy[0].weight)
    nn.init.kaiming_normal_(self.policy[3].weight)
    nn.init.xavier_normal_(self.policy[-1].weight)

    # value head

    self.value = nn.Sequential(
      nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(bottleneck_channels),
      nn.LeakyReLU(),
      nn.Conv2d(bottleneck_channels, 1, kernel_size=1, bias=False),
      nn.BatchNorm2d(1),
      nn.LeakyReLU(),
      nn.Flatten(),
      nn.Linear(board_size[0] * board_size[1], value_middle_width),
      nn.LeakyReLU(),
      nn.Dropout(0.5),
      nn.Linear(value_middle_width, 1),
      nn.Sigmoid(),
    )
    
    nn.init.kaiming_normal_(self.value[0].weight)
    nn.init.kaiming_normal_(self.value[3].weight)
    nn.init.kaiming_normal_(self.value[-5].weight)
    nn.init.xavier_normal_(self.value[-2].weight)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shared = self.shared(x)
    policy = self.policy(shared).squeeze(1) # remove channel dimension
    value = self.value(shared)
    return policy, value

  def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return self.forward(x)
