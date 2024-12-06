from .base import Encoder

from go.goboard import GameState, Point, Move
from go.gotypes import Player
from utils.eye_identifier import is_point_an_eye
from utils.ladder import analyze_ladder

import torch

__all__ = [
  'ZhuGoEncoder',
]


def iterable_points(game_state: GameState) -> list[Point]:
  return [
    Point(row + 1, col + 1)
    for row in range(game_state.board.num_rows) for col in range(game_state.board.num_cols)
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

  ENCODE_STRATEGY = {
  # encoder name            | plane count
    'encode_self_stone':      1,
    'encode_opponent_stone':  1,
    'encode_player':          1,
    'encode_valid_move':      1,
    'encode_qi':              8,
    'encode_qi_after_play':   8,
    'encode_ladder':          3,
    'encode_ko':              1,
  }

  def __init__(self, *, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
    super().__init__(device=device)

  def encode(self, game_state: GameState) -> torch.Tensor:
    planes = []
    for encoder_name in ZhuGoEncoder.ENCODE_STRATEGY.keys():
      planes.append(getattr(ZhuGoEncoder, encoder_name)(self, game_state))

    tensor = torch.cat(planes, dim=0).to(device=self.device)

    if __debug__:
      plane_count_sum = sum([plane_count for plane_count in ZhuGoEncoder.ENCODE_STRATEGY.values()])
      assert tensor.size() == (plane_count_sum, game_state.board.num_rows, game_state.board.num_cols)

    return tensor

  def encode_self_stone(self, game_state: GameState) -> torch.Tensor:
    tensor = torch.zeros((1, *game_state.board.size), device='cpu', dtype=torch.float32)

    for point in iterable_points(game_state):
      if game_state.board.get(point) == game_state.next_player:
        tensor[0, point.row - 1, point.col - 1] = 1.0

    return tensor

  def encode_opponent_stone(self, game_state: GameState) -> torch.Tensor:
    tensor = torch.zeros((1, *game_state.board.size), device='cpu', dtype=torch.float32)

    for point in iterable_points(game_state):
      if game_state.board.get(point) == game_state.next_player.other:
        tensor[0, point.row - 1, point.col - 1] = 1.0

    return tensor

  def encode_player(self, game_state: GameState) -> torch.Tensor:
    if game_state.next_player == Player.black:
      return torch.ones((1, *game_state.board.size), device='cpu', dtype=torch.float32)
    elif game_state.next_player == Player.white:
      return torch.zeros((1, *game_state.board.size), device='cpu', dtype=torch.float32)
    else:
      raise ValueError(f"Invalid player {game_state.next_player}")

  def encode_valid_move(self, game_state: GameState) -> torch.Tensor:
    tensor = torch.zeros((1, *game_state.board.size), device='cpu', dtype=torch.float32)

    for point in iterable_points(game_state):
      if game_state.is_valid_move(Move.play(point)) \
        and not is_point_an_eye(game_state.board, point, game_state.next_player):
        tensor[0, point.row - 1, point.col - 1] = 1.0

    return tensor

  def encode_qi(self, game_state: GameState) -> torch.Tensor:
    max_qi = 8

    tensor = torch.zeros((max_qi, *game_state.board.size), device='cpu', dtype=torch.float32)

    board = game_state.board
    for point in iterable_points(game_state):
      if board.get(point) is None:
        continue

      qi = board.qi(point)
      qi = min(max_qi, qi)

      if qi > 0:
        tensor[0 : qi, point.row - 1, point.col - 1] = 1.0

    return tensor

  def encode_qi_after_play(self, game_state: GameState) -> torch.Tensor:
    max_qi = 8

    tensor = torch.zeros((max_qi, *game_state.board.size), device='cpu', dtype=torch.float32)

    for point in iterable_points(game_state):
      if not game_state.is_valid_move(Move.play(point)):
        continue

      simulated_game_state = game_state.apply_move(Move.play(point))

      qi = simulated_game_state.board.qi(point)
      qi = min(max_qi, qi)

      if qi > 0:
        tensor[0 : qi, point.row - 1, point.col - 1] = 1.0

    return tensor

  def encode_ladder(self, game_state: GameState) -> torch.Tensor:
    tensor = torch.zeros((3, *game_state.board.size), device='cpu', dtype=torch.float32)

    ladder_analysis = analyze_ladder(game_state.board)

    for point in ladder_analysis.trapped_stones():
      player_offset = 0 if game_state.board.get(point) == game_state.next_player else 1
      tensor[player_offset, point.row - 1, point.col - 1] = 1.0

    for point in ladder_analysis.escape_paths():
      tensor[2, point.row - 1, point.col - 1] = 1.0

    return tensor

  def encode_ko(self, game_state: GameState) -> torch.Tensor:
    tensor = torch.zeros((1, *game_state.board.size), device='cpu', dtype=torch.float32)

    for point in iterable_points(game_state):
      if not game_state.board.is_valid_move(point, game_state.next_player):
        continue
      
      if game_state.does_move_violate_ko(game_state.next_player, Move.play(point)):
        tensor[0, point.row - 1, point.col - 1] = 1.0

    return tensor
