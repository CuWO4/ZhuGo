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

    # shuffle all moves, try linearly, and return the first valid one
    # if all moves are invalid, return pass turn

    board_size = game_state.board.num_rows * game_state.board.num_cols
    rand_indexes = list(range(0, board_size))
    random.shuffle(rand_indexes)
    for rand_idx in rand_indexes:
      rand_move = idx_to_move(rand_idx, game_state.board.size)
      if game_state.is_valid_move(rand_move) \
        and not is_point_an_eye(game_state.board, rand_move.point, game_state.next_player):
        return rand_move
    return goboard.Move.pass_turn()
