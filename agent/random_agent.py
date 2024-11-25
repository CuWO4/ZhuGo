import agent.base as base

import go.goboard as goboard
from agent.helpers import is_point_an_eye

import random

__all__ = [
  'RandomAgent'
]

class RandomAgent(base.Agent):
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
      