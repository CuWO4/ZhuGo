from go.goboard import GameState, Move
from ui.utils import MCTSData

import multiprocessing

__all__ = [
  'Agent'
]

class Agent:
  def __init__(self, *, need_move_queue: bool = False, need_mcts_queue: bool = False) -> None:
    self.need_move_queue: bool = need_move_queue
    self.need_mcts_queue: bool = need_mcts_queue
    self.move_queue: multiprocessing.Queue | None = None
    self.mcts_queue: multiprocessing.Queue | None = None
    
  def select_move(self, game_state: GameState) -> Move:
    raise NotImplementedError()
    
  def subscribe_move_queue(self, move_queue: multiprocessing.Queue):
    if self.need_move_queue:
      self.move_queue = move_queue
      
  def subscribe_mcts_queue(self, mcts_queue: multiprocessing.Queue):
    if self.need_mcts_queue:
      self.mcts_queue = mcts_queue

  def dequeue_move(self, turn_start_timestamp: int, game_state: GameState) -> Move:
    '''get last valid move after turn_start_timestamp(ms)'''
    if self.move_queue is None:
      return None
    
    last_valid_move = None
    while not self.move_queue.empty():
      move, time_stamp = self.move_queue.get()
      if time_stamp >= turn_start_timestamp and game_state.is_valid_move(move):
        last_valid_move = move
    return last_valid_move
      
  def enqueue_mcts_data(self, q: iter, visited_times: iter, best_idx: int | None, size: tuple[int, int]):
    '''q and visited_times indexing method should be consistent with utils.move_idx_transformer'''
    if self.mcts_queue is None:
      return
    self.mcts_queue.put(MCTSData(q, visited_times, best_idx, size))

  def enqueue_empty_mcts_data(self, size: tuple[int, int]):
    if self.mcts_queue is None:
      return
    self.mcts_queue.put(MCTSData.empty(size))
