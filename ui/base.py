from go.goboard import Move, GameState
import multiprocessing
import time

__all__ = [
  'UI'
]

class UI:
  def __init__(self, move_queue: multiprocessing.Queue, row_n: int = 19, col_n: int = 19):
    assert row_n == col_n
    self.move_queue = move_queue
    self.row_n = row_n
    self.col_n = col_n
  
  def update(self, game_state: GameState):
    raise NotImplementedError()
  
  @staticmethod
  def enqueue_move(move_queue: multiprocessing.Queue, move: Move):
    timestamp = int(time.time() * 1000)
    move_queue.put((move, timestamp))