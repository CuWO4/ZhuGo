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
    player                2     1/0                   all 1 if black, otherwise all 0 for first plane; all 0 if black otherwise 1 for
                                                      the second, to avoid bias introduce by zero padding
    valid move            1     1/0                   fill own eye is invalid
    qi(气)                8     1/0 ramp encoding     qi of certain stone string
    qi after play         8     1/0 ramp encoding     qi of certain stone string after play. 0 if invalid move
    ladder(征子)          3     1/0                   trapped points whose escape would trigger a ladder (self's and opponent's)
                                                      + laddering escaping path
    ko(劫争)              1     1/0                   1 if certain position is invalid because of ko
    position              1     continuous            how close the position is to the corner, nonlinear
    ----------------------------------------------
    sum                   26
  '''

  CHANNELS = 26

  SELF_STONE_OFF = 0
  OPPONENT_STONE_OFF = 1
  PLAYER_OFF = 2
  VALID_MOVE_OFF = 4
  QI_OFF = 5
  QI_AFTER_PLAY_OFF = 13
  LADDER_OFF = 21
  KO_OFF = 24
  POSITION_OFF = 25

  def __init__(self, *, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
    super().__init__(device=device)

  def encode(self, game_state: GameState) -> torch.Tensor:
    tensor = c_encode(game_state).to(device=self.device)

    assert tensor.size() == (self.CHANNELS, game_state.board.num_rows, game_state.board.num_cols)

    return tensor
