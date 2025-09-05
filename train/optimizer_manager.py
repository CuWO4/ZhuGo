from ai.zhugo import ZhuGo

import torch
import torch.optim as optim
import os

__all__ = [
  'OptimizerManager'
]


class OptimizerManager:
  OPTIMIZER_FILE = 'optimize.pth'

  def __init__(
    self, root: str, OptimizerType: type,
    *,
    shared_trunk_lr: float,
    policy_head_lr: float,
    value_head_lr: float,
    optim_args: tuple = (), optim_kwargs: dict = {},
  ):
    self.root = root
    self.shared_trunk_lr: float = shared_trunk_lr
    self.policy_head_lr: float = policy_head_lr
    self.value_head_lr: float = value_head_lr
    self.OptimizerType: type = OptimizerType
    self.optim_args: tuple = optim_args
    self.optim_kwargs: dict = optim_kwargs

  def save_optimizer(self, optimizer: optim.Optimizer):
    optimizer_path = os.path.join(self.root, self.OPTIMIZER_FILE)
    torch.save(optimizer.state_dict(), optimizer_path)
    print('optimizer saved successfully')

  def load_optimizer(self, model: ZhuGo):
    optimizer: optim.Optimizer = self.OptimizerType(
      [
        { 'params': model.shared.parameters(), 'lr': self.shared_trunk_lr },
        { 'params': model.policy.parameters(), 'lr': self.policy_head_lr },
        { 'params': model.value.parameters(), 'lr': self.value_head_lr },
      ],
      *self.optim_args, **self.optim_kwargs
    )

    optimizer_path = os.path.join(self.root, self.OPTIMIZER_FILE)
    try:
      state_dict = torch.load(optimizer_path)
      optimizer.load_state_dict(state_dict)
      print('optimizer loaded successfully')
    except Exception as e:
      print(f'failed to load optimizer state, use uninitialized optimizer instead (`{e}`)')

    return optimizer
