import torch
import torch.nn as nn

__all__ = [
  'ZhuGo',
]

class ResidualConvBlock(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(ResidualConvBlock, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(output_channels),
      nn.LeakyReLU(),
      nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(output_channels),
    )

  def forward(self, x):
    return nn.functional.leaky_relu(self.model(x) + x)

class MultiScaleConvBlock(nn.Module):
  def __init__(self, input_channels, output_channels, bias=True):
    super(MultiScaleConvBlock, self).__init__()
    channels7x7 = output_channels // 8
    channels5x5 = output_channels // 4
    channels3x3 = output_channels - channels7x7 - channels5x5

    self.conv3x3 = nn.Conv2d(input_channels, channels3x3, kernel_size=3, padding=1, bias=bias)
    self.conv5x5 = nn.Conv2d(input_channels, channels5x5, kernel_size=5, padding=2, bias=bias)
    self.conv7x7 = nn.Conv2d(input_channels, channels7x7, kernel_size=7, padding=3, bias=bias)

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
      residual_layers += [ResidualConvBlock(channel, channel) for _ in range(depth)]
      if idx < length - 1:
        next_channel = residual_channels[idx + 1]
        residual_layers += [
          MultiScaleConvBlock(channel, next_channel, bias=False),
          nn.BatchNorm2d(next_channel),
          nn.LeakyReLU(),
          nn.Dropout2d(0.2),
        ]

    first_channel = residual_channels[0]
    last_channel = residual_channels[-1]

    self.shared = nn.Sequential(
      # input layers
      nn.Conv2d(input_channels, first_channel, kernel_size=5, padding=2, bias=False),
      nn.BatchNorm2d(first_channel),
      nn.LeakyReLU(),

      # residual layers
      *residual_layers,

      # Bottleneck convolution
      nn.Conv2d(last_channel, bottleneck_channels, kernel_size=1, bias=False),
      nn.BatchNorm2d(bottleneck_channels),
      nn.LeakyReLU(),
    )

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

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shared = self.shared(x)
    policy = self.policy(shared).squeeze(1) # remove channel dimension
    value = self.value(shared)
    return policy, value

  def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return self.forward(x)
