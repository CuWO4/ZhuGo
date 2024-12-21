from .base import Encoder

from go.goboard import GameState

from cencoder.cencoder import c_encode

import torch

__all__ = [
  'ZhuGoEncoder',
]

class ZhuGoEncoder(Encoder):
  '''
  input planes:
    self's stone          1     1/0
    opponent's stone      1     1/0
    player                1     1/0                   all 1 if black, otherwise all 0
    valid move            1     1/0                   fill own eye is invalid
    qi(气)                8     1/0 ramp encoding     qi of certain stone string
    qi after play         8     1/0 ramp encoding     qi of certain stone string after play. 0 if invalid move
    ladder(征子)          3     1/0                   trapped points whose escape would trigger a ladder (self's and opponent's)
                                                      + laddering escaping path
    ko(劫争)              1     1/0                   1 if certain position is invalid because of ko
    ----------------------------------------------
    sum                   24
  '''

  CHANNELS = 24

  def __init__(self, *, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
    super().__init__(device=device)

  def encode(self, game_state: GameState) -> torch.Tensor:
    tensor = c_encode(game_state).cuda()

    assert tensor.size() == (self.CHANNELS, game_state.board.num_rows, game_state.board.num_cols)

    return tensor
