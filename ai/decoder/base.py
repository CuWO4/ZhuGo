from go.goboard import Move

import torch

__all__ = [
  'Decoder',
]

class Decoder:
  def __init__(self, size: tuple[int, int], *, device: str) -> None:
    '''size is tuple (rows, cols)'''
    self.size: tuple[int, int] = size
    self.device = device

  def decode(self, policy_tensor: torch.Tensor) -> Move:
    '''policy tensor is in (rows, cols) shape
    '''
    raise NotImplementedError()