import agent.base as base

import go.goboard as goboard

import time
import multiprocessing

__all__ = [
  'HumanAgent'
]

class HumanAgent(base.Agent):
  def __init__(self, move_queue: multiprocessing.Queue = None):
    self.move_queue = move_queue
    
  def select_move(self, game_state: goboard.GameState) -> goboard.Move:
    assert self.move_queue is not None
    
    turn_start_timestamp = int(time.time() * 1000)
    while True:
      move = self.dequeue_move(turn_start_timestamp)
      if move is not None and game_state.is_valid_move(move):
        break
      time.sleep(0.01)

    return move
  
  def dequeue_move(self, turn_start_timestamp: int) -> goboard.Move | None:
    last_move = None
    while not self.move_queue.empty():
      move, time_stamp = self.move_queue.get()
      if time_stamp >= turn_start_timestamp:
        last_move = move
    return last_move

  def subscribe_move_queue(self, move_queue: multiprocessing.Queue):
    self.move_queue = move_queue