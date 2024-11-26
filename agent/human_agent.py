from agent.base import Agent
from go.goboard import GameState, Move

import time
import multiprocessing

__all__ = [
  'HumanAgent'
]

class HumanAgent(Agent):
  def __init__(self, *, need_move_queue: bool = True, need_mcts_queue: bool = False):
    super().__init__(need_move_queue=need_move_queue, need_mcts_queue=need_mcts_queue)
    
  def select_move(self, game_state: GameState) -> Move:
    assert self.move_queue is not None
    
    turn_start_timestamp = int(time.time() * 1000)
    while True:
      move = self.dequeue_move(turn_start_timestamp)
      if move is not None and game_state.is_valid_move(move):
        break
      time.sleep(0.05)

    return move
