'''
indexings are consistent with utils.move_idx_transformer
'''

import numpy as np

__all__ = [
  'exploring_move_indexes',
  'best_move_idx',
  'cal_entropy'
]

def exploring_move_indexes(ucb: np.ndarray[float], size: int) -> list[int]:
  max_ucb_indexes = np.argwhere(ucb >= np.max(ucb) - 1e-3).flatten()
  
  if len(max_ucb_indexes) == 0:
    print(f'{max_ucb_indexes=}')
    assert len(max_ucb_indexes)
    
  move_indexes = np.random.choice(max_ucb_indexes, size=size, replace=True)
  return move_indexes

def best_move_idx(
  visited_times: np.ndarray[int],
  q: np.ndarray
) -> int:
  max_visited_indexes = np.argwhere(visited_times == np.max(visited_times)).flatten()

  if len(max_visited_indexes) == 0:
    print(f'{max_visited_indexes=}')
    assert len(max_visited_indexes)

  max_q_idx_in_max_visited = np.argmax(q[max_visited_indexes])
  max_idx = max_visited_indexes[max_q_idx_in_max_visited]
  return max_idx

def cal_entropy(array: np.ndarray) -> float:
  '''array do not need to be normalized'''
  distribution = (array + 1e-8) / np.sum(array + 1e-8)
  return - np.sum(distribution * np.log2(distribution))