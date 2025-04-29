from .boardmodule import (
  new_board,
  delete_board,
  clone_board,
  get_piece,
  get_qi,
  get_random_qi_pos,
  is_valid_move,
  place_piece,
  hash as zobrist_hash,
  serialize,
  deserialize,
  does_violate_ko as does_violate_ko_c,
)
# visit C extension functions by `boardmodule.xxx` may fail and get None when torch.nn is imported
# before this file for unknown reason, while `from ... import ...` would not trigger the bug

__all__ = [
  'cBoard',
  'does_violate_ko'
]

class cBoard:
  def __init__(self, rows: int, cols: int, c_board = None):
    self.rows = rows
    self.cols = cols
    if c_board is None:
      self._c_board = new_board(rows, cols)
    else:
      self._c_board = c_board

  def __deepcopy__(self, memo={}):
    return cBoard(self.rows, self.cols, clone_board(self._c_board))

  def get(self, row: int, col: int) -> int:
    return get_piece(self._c_board, row, col)

  def qi(self, row: int, col: int) -> int:
    return get_qi(self._c_board, row, col)

  def get_random_qi_pos(self, row: int, col: int) -> tuple[int, int]:
    return get_random_qi_pos(self._c_board, row, col)

  def is_valid_move(self, player: int, row: int, col: int) -> bool:
    return is_valid_move(self._c_board, row, col, player)

  def place_stone(self, player: int, row: int, col: int):
    place_piece(self._c_board, row, col, player)

  def hash(self) -> int:
    return zobrist_hash(self._c_board)

  def to_board_data(self):
    return [
      [self.get(row, col) for col in range(self.cols)]
      for row in range(self.rows)
    ]

  @staticmethod
  def from_board_data(borad_data):
    rows = len(borad_data)
    cols = len(borad_data[0])

    c_board = new_board(rows, cols)
    for r, row in enumerate(borad_data):
      for c, player in enumerate(row):
        place_piece(c_board, r, c, player)
    return c_board

  def __getstate__(self) -> dict:
    state = self.__dict__.copy()
    state['_c_board'] = serialize(self._c_board)
    return state

  def __setstate__(self, state: dict):
    self.__dict__.update(state)
    self._c_board = deserialize(self._c_board)

def does_violate_ko(board: cBoard, player: int, row: int, col: int, last_board: cBoard) -> bool:
  return does_violate_ko_c(board._c_board, player, row, col, last_board._c_board)