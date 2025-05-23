from go.goboard import GameState

import torch

__all__ = [
  'Encoder',
]

class Encoder:
  def __init__(self, *, device: str) -> None:
    self.device = device

  def encode(self, game_state: GameState) -> torch.Tensor:
    '''return (channels, rows, cols) shape Tensor'''
    raise NotImplementedError()