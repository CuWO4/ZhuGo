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

class IntermediateConvBlock(nn.Module):
  def __init__(self, input_channels, output_channels, bias=True):
    super(IntermediateConvBlock, self).__init__()
    self.conv3x3 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=bias)
    nn.init.kaiming_normal_(self.conv3x3.weight)

  def forward(self, x):
    return self.conv3x3(x)

class ZhuGo(nn.Module):
  def __init__(
    self, *,
    board_size: tuple[int, int],
    input_channels : int,
    # note  each depth of residual block contain 2 convolution layers
    residual_channels: tuple[int], 
    residual_depths: tuple[int], 
    bottleneck_channels: int,
    policy_residual_depth: int,
    value_middle_width: int
  ):
    super(ZhuGo, self).__init__()

    # shared

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
        residual_layers += [
          nn.Dropout2d(0.1),
          IntermediateConvBlock(channel, next_channel, bias=False),
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
    )
    
    nn.init.kaiming_normal_(self.shared[0].weight)
    nn.init.kaiming_normal_(self.shared[-1].weight)

    # policy head

    self.policy = nn.Sequential(
      *[
        ZhuGoResidualConvBlock(bottleneck_channels)
        for _ in range(policy_residual_depth)
      ],
      nn.BatchNorm2d(bottleneck_channels),
      nn.LeakyReLU(),
      nn.Conv2d(bottleneck_channels, 1, kernel_size=1)
    )
    
    nn.init.xavier_normal_(self.policy[-1].weight)

    # value head

    self.value = nn.Sequential(
      ZhuGoResidualConvBlock(bottleneck_channels),
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
    )
    
    nn.init.kaiming_normal_(self.value[3].weight)
    nn.init.kaiming_normal_(self.value[-4].weight)
    nn.init.xavier_normal_(self.value[-1].weight)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shared = self.shared(x)
    policy = self.policy(shared).squeeze(1) # remove channel dimension
    value = self.value(shared)
    return policy, value

  def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return self.forward(x)
