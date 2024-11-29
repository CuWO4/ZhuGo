from agent.base import Agent
from go.goboard import GameState, Move

import time

__all__ = [
  'HumanAgent'
]

class HumanAgent(Agent):
  def __init__(self, *, need_move_queue: bool = True, need_mcts_queue: bool = False):
    super().__init__(need_move_queue=need_move_queue, need_mcts_queue=need_mcts_queue)
    
  def select_move(self, game_state: GameState) -> Move:
    assert self.move_queue is not None
    
    turn_start_timestamp = time.time()
    while True:
      move = self.dequeue_move(turn_start_timestamp, game_state)
      if move is not None:
        return move
      time.sleep(0.05)
