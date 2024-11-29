from .base import Agent

import go.goboard as goboard
from utils.eye_identifier import is_point_an_eye

import random

__all__ = [
  'RandomAgent'
]

class RandomAgent(Agent):
  def __init__(self, *, need_move_queue: bool = True, need_mcts_queue: bool = False):
    super().__init__(need_move_queue=need_move_queue, need_mcts_queue=need_mcts_queue)
  
  def select_move(self, game_state: goboard.GameState) -> goboard.Move:
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
      