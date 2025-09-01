import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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
  def __init__(self, channels, *, checkpoint: bool):
    super(ResidualConvBlock, self).__init__()
    self.model = nn.Sequential(
      nn.BatchNorm2d(channels),
      nn.GELU(),
      nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(channels),
      nn.GELU(),
      nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
    )

    kaiming_init_sequential(self.model)

    self.checkpoint = checkpoint

  def forward(self, x):
    return checkpoint(self.model, x, use_reentrant = False) + x if self.checkpoint else self.model(x) + x

class GlobalBiasBLock(nn.Module):
  '''(B, C, N, M) -> (B, C, N, M)'''
  def __init__(self, channel, *, checkpoint: bool):
    super(GlobalBiasBLock, self).__init__()
    self.activate = nn.Sequential(
      nn.BatchNorm2d(channel),
      nn.GELU(),
    )

    self.linear = nn.Linear(2 * channel, channel)

    nn.init.xavier_normal_(self.linear.weight)
    nn.init.zeros_(self.linear.bias)

    self.checkpoint = checkpoint

  def forward(self, x: torch.Tensor):
    y = checkpoint(self.activate, x, use_reentrant = False) if self.checkpoint else self.activate(x)
    plane_means = y.mean(dim = (-2, -1)).flatten(start_dim = 1)
    plane_maxes = y.amax(dim = (-2, -1)).flatten(start_dim = 1)
    y = checkpoint(self.linear, torch.cat((plane_means, plane_maxes), dim = 1), use_reentrant = False) \
      if self.checkpoint else self.linear(torch.cat((plane_means, plane_maxes), dim = 1))
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
  def __init__(self, channels, *, checkpoint: bool):
    super(ZhuGoResidualConvBlock, self).__init__()

    inner_channels = channels // 2

    self.encoder_conv1x1 = nn.Sequential(
      nn.BatchNorm2d(channels),
      nn.GELU(),
      nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False),
    )

    self.inner_residual_blocks = nn.Sequential(
      ResidualConvBlock(inner_channels, checkpoint = checkpoint),
      ResidualConvBlock(inner_channels, checkpoint = checkpoint),
    )

    self.decoder_conv1x1 = nn.Sequential(
      nn.BatchNorm2d(inner_channels),
      nn.GELU(),
      nn.Conv2d(inner_channels, channels, kernel_size=1, bias=False)
    )

    kaiming_init_sequential(self.encoder_conv1x1)
    kaiming_init_sequential(self.decoder_conv1x1)

    self.checkpoint = checkpoint

  def forward(self, x):
    out = checkpoint(self.encoder_conv1x1, x, use_reentrant = False) \
      if self.checkpoint else self.encoder_conv1x1(x)
    out = self.inner_residual_blocks(out)
    out = checkpoint(self.decoder_conv1x1, out, use_reentrant = False) + x \
      if self.checkpoint else self.decoder_conv1x1(out) + x
    return out

class MeanMax2dGlobalPool(nn.Module):
  '''(B, C, N, M) -> (B, 2 * C, 1, 1)'''
  def __init__(self):
    super(MeanMax2dGlobalPool, self).__init__()

  def forward(self, x):
    return torch.cat((
      torch.mean(x, dim = (-2, -1), keepdim = True),
      torch.amax(x, dim = (-2, -1), keepdim = True),
    ), dim = 1)

class ZhuGoSharedResNet(nn.Module):
  '''(B, C1, N, M) -> (B, C2, N, M)'''
  def __init__(
    self,
    input_channels: int,
    residual_channel: int,
    residual_depth: int,
    *,
    checkpoint: bool
  ):
    super(ZhuGoSharedResNet, self).__init__()

    residual_layers = []
    total_depth = 0
    for _ in range(residual_depth):
      residual_layers.append(ZhuGoResidualConvBlock(residual_channel, checkpoint = checkpoint))
      if total_depth % 3 == 2:
        residual_layers.append(GlobalBiasBLock(residual_channel, checkpoint = checkpoint))
      total_depth += 1

    self.model = nn.Sequential(
      nn.Conv2d(input_channels, residual_channel, kernel_size=5, padding=2, bias=False),
      *residual_layers,
    )

    kaiming_init_sequential(self.model)

  def forward(self, x):
    return self.model(x)

# https://github.com/lightvector/KataGo/blob/6260e62c66132bd3d37e894ede213858324db45d/python/model_pytorch.py
class ZhuGoPolicyHead(nn.Module):
  '''(B, C, N, M) -> (B, N * M + 1)'''
  def __init__(
    self,
    residual_channel: int,
    *,
    checkpoint: bool
  ):
    super(ZhuGoPolicyHead, self).__init__()

    self.outg = nn.Sequential(
      nn.BatchNorm2d(residual_channel),
      nn.Conv2d(residual_channel, residual_channel, 1, bias = False),
      nn.BatchNorm2d(residual_channel),
      nn.GELU(),
      MeanMax2dGlobalPool(),
      nn.Flatten()
    )

    self.pass_linear = nn.Sequential(
      nn.Linear(2 * residual_channel, residual_channel // 2),
      nn.GELU(),
      nn.Linear(residual_channel // 2, 1)
    )

    self.assist_linear = nn.Linear(2 * residual_channel, residual_channel)

    self.pre_outp = nn.Sequential(
      nn.BatchNorm2d(residual_channel),
      nn.Conv2d(residual_channel, residual_channel, 1, bias = False),
      nn.BatchNorm2d(residual_channel),
      nn.GELU(),
    )

    self.post_outp = nn.Sequential(
      nn.BatchNorm2d(residual_channel),
      nn.GELU(),
      nn.Conv2d(residual_channel, 1, 1)
    )

    kaiming_init_sequential(self.outg)
    kaiming_init_sequential(self.pass_linear)
    nn.init.xavier_normal_(self.pass_linear[-1].weight)
    nn.init.zeros_(self.pass_linear[-1].bias)
    nn.init.kaiming_normal_(self.assist_linear.weight)
    nn.init.zeros_(self.assist_linear.bias)
    kaiming_init_sequential(self.pre_outp)
    kaiming_init_sequential(self.post_outp)
    nn.init.xavier_normal_(self.post_outp[-1].weight)
    nn.init.zeros_(self.post_outp[-1].bias)

    self.checkpoint = checkpoint

  def forward(self, x):
    outg = checkpoint(self.outg, x, use_reentrant = False) \
      if self.checkpoint else self.outg(x) # (B, 2 * C)
    pass_logits = self.pass_linear(outg)
    outg = checkpoint(self.assist_linear, outg, use_reentrant = False) \
      if self.checkpoint else self.assist_linear(outg)
    outg = outg.unsqueeze(-1).unsqueeze(-1)
    outp = checkpoint(self.pre_outp, x, use_reentrant = False) \
      if self.checkpoint else self.pre_outp(x)
    outp = outp + outg
    outp = checkpoint(self.post_outp, outp, use_reentrant = False) \
      if self.checkpoint else self.post_outp(outp)
    return torch.cat((
      torch.flatten(outp, start_dim = 1),
      pass_logits
    ), dim = 1)

class ZhuGoValueHead(nn.Module):
  '''
  (B, C, N, M) -> (B, 1), (B, N * M), (B, SCORE_BUCKET_COUNT)
  win_rate, ownership, score
  '''
  # lower bound, upper bound, all buckets contains lower bound
  SCORE_BUCKETS = torch.tensor((
    (-1, -0.3),
    (-0.3, -0.1),
    (-0.1, -0.03),
    (-0.03, -0.01),
    (-0.01, 0),
    (0, 0.01),
    (0.01, 0.03),
    (0.03, 0.1),
    (0.1, 0.3),
    (0.3, 1),
  ), device = 'cuda')
  def __init__(
    self,
    residual_channel: int,
    value_middle_width: int,
    *,
    checkpoint: bool
  ):
    super(ZhuGoValueHead, self).__init__()

    self.pre_gpool = nn.Sequential(
      nn.BatchNorm2d(residual_channel),
      nn.GELU(),
      nn.Conv2d(residual_channel, residual_channel, 1, bias=False),
      nn.BatchNorm2d(residual_channel),
      nn.GELU(),
    )

    self.gpool_flatten_linear = nn.Sequential(
      MeanMax2dGlobalPool(),
      nn.Flatten(),
      nn.Linear(2 * residual_channel, value_middle_width),
      nn.GELU(),
    )

    self.win_rate_head = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(value_middle_width, 1),
    )

    self.score_head = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(value_middle_width, self.SCORE_BUCKETS.shape[0]),
    )

    self.ownership_head = nn.Sequential(
      nn.Conv2d(residual_channel, 1, 1, bias=True),
      nn.Flatten(),
    )

    kaiming_init_sequential(self.pre_gpool)
    kaiming_init_sequential(self.gpool_flatten_linear)
    kaiming_init_sequential(self.win_rate_head)
    kaiming_init_sequential(self.score_head)

    nn.init.xavier_normal_(self.win_rate_head[-1].weight)
    nn.init.zeros_(self.win_rate_head[-1].bias)
    nn.init.xavier_normal_(self.score_head[-1].weight)
    nn.init.zeros_(self.score_head[-1].bias)
    nn.init.xavier_normal_(self.ownership_head[0].weight)
    nn.init.zeros_(self.ownership_head[0].bias)

    self.checkpoint = checkpoint

  def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if self.checkpoint:
      shared = checkpoint(self.pre_gpool, x, use_reentrant=False)
      ownership = checkpoint(self.ownership_head, shared, use_reentrant=False)
      shared = checkpoint(self.gpool_flatten_linear, shared, use_reentrant=False)
      win_rate = checkpoint(self.win_rate_head, shared, use_reentrant=False)
      score = checkpoint(self.score_head, shared, use_reentrant=False)
    else:
      shared = self.pre_gpool(x)
      ownership = self.ownership_head(shared)
      shared = self.gpool_flatten_linear(shared)
      win_rate = self.win_rate_head(shared)
      score = self.score_head(shared)
    return win_rate, ownership, score

class ZhuGo(nn.Module):
  '''
  (B, C, N, M) -> (B, N * M + 1), (B, 1), (B, N * M), (B, SCORE_BUCKET_COUNT)
  first is policy output logits, unnormalized, 361 (row-major moves) + 1 (pass turn).
  second is win_rate output logits, inactivated.
  third is ownership prediction logits, inactivated & unnormalized.
  fourth is scoring prediction logits, being consistent with ZhuGoValueHead.SCORE_BUCKETS, inactivated.
  '''
  def __init__(
    self, *,
    board_size: tuple[int, int],
    input_channels : int,
    residual_channels: tuple[int],
    residual_depths: tuple[int],
    bottleneck_channels: int | None = None,
    policy_residual_depth: int | None = None,
    value_residual_depth: int | None = None,
    value_middle_width: int,
    checkpoint: bool = True
  ):
    super(ZhuGo, self).__init__()

    if len(residual_channels) > 1:
      print('runtime warning: multi residual channels is no longer supported, only first is used, '
            'others ignored...')
    if bottleneck_channels is not None:
      print('runtime warning: bottleneck_channels is deprecated, ignored...')
    if policy_residual_depth is not None:
      print('runtime warning: policy_residual_depth is deprecated, ignored...')
    if value_residual_depth is not None:
      print('runtime warning: value_residual_depth is deprecated, ignored...')

    self.shared = ZhuGoSharedResNet(
      input_channels, residual_channels[0], residual_depths[0], checkpoint = checkpoint
    )

    self.policy = ZhuGoPolicyHead(
      residual_channels[0], checkpoint = checkpoint
    )

    self.value = ZhuGoValueHead(
      residual_channels[0], value_middle_width, checkpoint = checkpoint
    )

  def forward(self, x: torch.Tensor):
    shared = self.shared(x)
    policy = self.policy(shared)
    win_rate, ownership, score = self.value(shared)
    return policy, win_rate, ownership, score
