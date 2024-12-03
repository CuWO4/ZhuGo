from go.goboard import Move
from go.gotypes import Point

__all__ = [
  'move_to_idx',
  'idx_to_move'
]

def move_to_idx(move: Move, size: tuple[int, int]) -> int:
  row_n, col_n = size
  if move.is_play:
    idx = (move.point.row - 1) * col_n + (move.point.col - 1)
  elif move.is_pass:
    idx = row_n * col_n
  elif move.is_resign:
    idx = row_n * col_n + 1
  else:
    assert False

  assert 0 <= idx < row_n * col_n + 2
  return idx

def idx_to_move(idx: int, size: tuple[int, int]) -> Move:
  row_n, col_n = size
  assert 0 <= idx < row_n * col_n + 2
  if idx >= row_n * col_n + 1:
    return Move.resign()
  elif idx >= row_n * col_n:
    return Move.pass_turn()
  else:
    return Move.play(Point(row = idx // col_n + 1, col = idx % col_n + 1))
  

def test() -> None:
  '''
  >>> move_to_idx(idx_to_move(0, (19, 19)), (19, 19)) == 0 \
  and move_to_idx(idx_to_move(360, (19, 19)), (19, 19)) == 360 \
  and move_to_idx(idx_to_move(361, (19, 19)), (19, 19)) == 361 \
  and move_to_idx(idx_to_move(362, (19, 19)), (19, 19)) == 362 \
  and move_to_idx(idx_to_move(265, (19, 19)), (19, 19)) == 265 \
  and move_to_idx(idx_to_move(265, (19, 17)), (19, 17)) == 265 \
  and move_to_idx(idx_to_move(265, (17, 19)), (17, 19)) == 265
  True
  '''
  