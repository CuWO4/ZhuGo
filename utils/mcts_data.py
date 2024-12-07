from go.goboard import Move
from go.goboard import Point
from .move_idx_transformer import move_to_idx, idx_to_move

__all__ = [
  'MCTSData'
]

class MCTSData:
  def __init__(self, q: iter, visited_times: iter, best_idx: int | None, win_rate: float | None, size: tuple[int, int]) -> None:
    self.q = q
    self.visited_times: iter = visited_times
    self.best_idx: int | None = best_idx
    self.win_rate: float | None = win_rate
    self.size: tuple[int, int] = size
    
  @staticmethod
  def empty(size: tuple[int, int]):
    '''return empty MCTSData'''
    row_n, col_n = size
    policy_size = row_n * col_n + 2
    return MCTSData([0] * policy_size, [0] * policy_size, None, None, size)
    
  def get(self, *, row: int, col: int) -> tuple[float, int]:
    '''return (q, visited_time). index starts from 0.'''
    idx = move_to_idx(Move.play(Point(row = row + 1, col = col + 1)), self.size)
    return self.q[idx], self.visited_times[idx]
  
  def best_pos(self) -> tuple[int, int] | None:
    '''return current best move position if it's a play otherwise None. index starts from 0.'''
    if self.best_idx is None or self.best_idx >= self.size[0] * self.size[1]:
      return None
    
    best_move_point = idx_to_move(self.best_idx, self.size).point
    return best_move_point.row - 1, best_move_point.col - 1