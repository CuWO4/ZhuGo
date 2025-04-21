import torch
import torch.nn as nn
import math

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

class ParamAttentionMixing(nn.Module):
  def __init__(self, min_alpha: float = 0.2, max_alpha: float = 0.8):
    super(ParamAttentionMixing, self).__init__()
    self.min_alpha = min_alpha
    self.max_alpha = max_alpha
    self.alpha = nn.Parameter(torch.tensor(0.0))

  def forward(self, x, atten):
    alpha = torch.sigmoid(self.alpha) * (self.max_alpha - self.min_alpha) + self.min_alpha

    # (1 - alpha) * x + alpha * x * atten
    return x + alpha * x * (atten - 1)

class TwoWayECABlock(nn.Module):
  '''(B, C, N, M) -> (B, C, N, M)'''
  def __init__(self, channels: int, gamma: float = 2.0, beta: float = 1.0):
    super(TwoWayECABlock, self).__init__()

    k_size = max(int((math.log2(channels) + beta) / gamma), 1)
    k_size = k_size if k_size % 2 else k_size + 1  # ensure odd

    self.atten = nn.Sequential(
      nn.Conv1d(2, 1, kernel_size = k_size, padding = (k_size - 1) // 2, bias = False),
      nn.BatchNorm1d(1),
      nn.Sigmoid(),
    )

    self.param_attention_mixing = ParamAttentionMixing()

    nn.init.xavier_normal_(self.atten[0].weight)

  def forward(self, x):
    avg = x.mean((-2, -1)) # (B, C, N, M) -> (B, C)
    max = x.amax((-2, -1)) # (B, C, N, M) -> (B, C)
    atten = self.atten(torch.stack((avg, max), dim = 1)) # (B, 1, C)
    atten = atten.permute(0, 2, 1).unsqueeze(-1) # (B, 1, C) -> (B, C, 1, 1)
    return self.param_attention_mixing(x, atten)

class CBAMBlock(nn.Module):
  '''(B, C, H, W) -> (B, C, H, W)'''
  def __init__(self, channels: int, reduction: int = 16):
    super(CBAMBlock, self).__init__()
    bottleneck_channels = channels // reduction

    self.channel_avg_fc = nn.Sequential(
      nn.Linear(channels, bottleneck_channels),
      nn.LeakyReLU()
    )
    self.channel_max_fc = nn.Sequential(
      nn.Linear(channels, bottleneck_channels),
      nn.LeakyReLU()
    )
    self.channel_shared_fc = nn.Sequential(
      nn.Linear(bottleneck_channels, channels),
      nn.Sigmoid()
    )

    self.plane_atten = nn.Sequential(
      nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
      nn.BatchNorm2d(1),
      nn.Sigmoid()
    )

    self.param_attention_mixing = ParamAttentionMixing()

    nn.init.kaiming_normal_(self.channel_avg_fc[0].weight, nonlinearity='leaky_relu', a=0.01)
    nn.init.zeros_(self.channel_avg_fc[0].bias)
    nn.init.kaiming_normal_(self.channel_max_fc[0].weight, nonlinearity='leaky_relu', a=0.01)
    nn.init.zeros_(self.channel_max_fc[0].bias)
    nn.init.xavier_normal_(self.channel_shared_fc[0].weight)
    nn.init.zeros_(self.channel_shared_fc[0].bias)

    nn.init.xavier_normal_(self.plane_atten[0].weight)

  def forward(self, x):
    avg = torch.mean(x, dim = (-2, -1))  # (B, C)
    avg = self.channel_avg_fc(avg)

    max = torch.amax(x, dim = (-2, -1))  # (B, C)
    max = self.channel_max_fc(max)

    channel_atten = self.channel_shared_fc(avg + max).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

    avg_plane = torch.mean(x, dim = 1, keepdim = True)  # (B, 1, H, W)
    max_plane = torch.amax(x, dim = 1, keepdim = True)  # (B, 1, H, W)
    plane_concat = torch.cat([avg_plane, max_plane], dim = 1)  # (B, 2, H, W)
    plane_atten = self.plane_atten(plane_concat)  # (B, 1, H, W)

    return self.param_attention_mixing(x, channel_atten * plane_atten)

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
#   |   ECA
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
      TwoWayECABlock(inner_channels),
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
      residual_layers += [ZhuGoResidualConvBlock(channel) for _ in range(depth)]
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

    self.shared = nn.Sequential(
      *[
        ZhuGoResidualConvBlock(bottleneck_channels)
        for _ in range(policy_residual_depth)
      ],
      nn.BatchNorm2d(bottleneck_channels),
      CBAMBlock(bottleneck_channels, reduction = 8),
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

    self.residual = nn.Sequential(
      *[
        ZhuGoResidualConvBlock(bottleneck_channels)
        for _ in range(value_residual_depth)
      ],
      nn.BatchNorm2d(bottleneck_channels),
      CBAMBlock(bottleneck_channels, reduction = 8),
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
