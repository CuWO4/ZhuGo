from .base import Agent

import go.goboard as goboard
from utils.eye_identifier import is_point_an_eye
from utils.move_idx_transformer import idx_to_move

import random

__all__ = [
  'RandomAgent'
]

class RandomAgent(Agent):
  def __init__(self, *, need_move_queue: bool = True, need_mcts_queue: bool = False):
    super().__init__(need_move_queue=need_move_queue, need_mcts_queue=need_mcts_queue)
  
  def select_move(self, game_state: goboard.GameState) -> goboard.Move:

    # try random actions first and return the first legal one
    # after failing several times, randomly select from all legal actions
    board_size = game_state.board.num_rows * game_state.board.num_cols
    for _ in range(board_size // 4):
      rand_idx = random.randrange(0, board_size)
      rand_move = idx_to_move(rand_idx, game_state.board.size)
      if game_state.is_valid_move(rand_move) \
        and not is_point_an_eye(game_state.board, rand_move.point, game_state.next_player):
        return rand_move

    candidates = []
    for move in game_state.legal_moves():
      if not move.is_pass and \
        not move.is_resign and \
        not is_point_an_eye(game_state.board, move.point, game_state.next_player):
        candidates.append(move)
    if not candidates:
      return goboard.Move.pass_turn()
    else:
      return random.choice(candidates)
      