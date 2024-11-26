from go.goboard import Move, GameState
import multiprocessing
import time

__all__ = [
  'UI'
]

class UI:
  def __init__(self, move_queue: multiprocessing.Queue, mcts_queue: multiprocessing.Queue, row_n: int = 19, col_n: int = 19):
    assert row_n == col_n
    self.move_queue: multiprocessing.Queue = move_queue
    self.mcts_queue: multiprocessing.Queue = mcts_queue
    self.row_n: int = row_n
    self.col_n: int = col_n
  
  def update(self, game_state: GameState):
    raise NotImplementedError()
  
  @staticmethod
  def enqueue_move(move_queue: multiprocessing.Queue, move: Move):
    timestamp = time.time()
    move_queue.put((move, timestamp))