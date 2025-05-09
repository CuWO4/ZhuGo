import torch
import torch.optim as optim
import os
from torch.nn.parameter import Parameter
from typing import Iterable

class OptimizerManager:
  OPTIMIZER_FILE = 'optimize.pth'

  def __init__(
    self, root: str, OptimizerType: type, 
    *, optim_args: tuple = (), optim_kwargs: dict = {},
  ):
    self.root = root
    self.OptimizerType: type = OptimizerType
    self.optim_args: tuple = optim_args
    self.optim_kwargs: dict = optim_kwargs

  def save_optimizer(self, optimizer: optim.Optimizer):
    optimizer_path = os.path.join(self.root, self.OPTIMIZER_FILE)
    torch.save(optimizer.state_dict(), optimizer_path)
    print('optimizer saved successfully')

  def load_optimizer(self, model_parameters: Iterable[Parameter]):
    optimizer: optim.Optimizer = self.OptimizerType(
      model_parameters, *self.optim_args, **self.optim_kwargs
    )

    optimizer_path = os.path.join(self.root, self.OPTIMIZER_FILE)
    if os.path.exists(optimizer_path):
      state_dict = torch.load(optimizer_path)
      optimizer.load_state_dict(state_dict)
      print('optimizer loaded successfully')
    else:
      print('failed to load optimizer state, use uninitialized optimizer instead')

    return optimizer
