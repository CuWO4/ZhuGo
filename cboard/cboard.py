from . import boardmodule

class cBoard:
  def __init__(self, rows: int, cols: int, c_board = None):
    self.rows = rows
    self.cols = cols
    if c_board is None:
      self._c_board = boardmodule.new_board(rows, cols)
    else:
      self._c_board = c_board

  def __del__(self):
    boardmodule.delete_board(self._c_board)

  def __deepcopy__(self, memo={}):
    return cBoard(self.rows, self.cols, boardmodule.clone_board(self._c_board))

  def get(self, row: int, col: int) -> int:
    return boardmodule.get_piece(self._c_board, row, col)

  def qi(self, row: int, col: int) -> int:
    return boardmodule.get_qi(self._c_board, row, col)
  
  def get_random_qi_pos(self, row: int, col: int) -> tuple[int, int]:
    return boardmodule.get_random_qi_pos(self._c_board, row, col)

  def is_valid_move(self, player: int, row: int, col: int) -> bool:
    return boardmodule.is_valid_move(self._c_board, row, col, player)

  def place_stone(self, player: int, row: int, col: int):
    boardmodule.place_piece(self._c_board, row, col, player)

  def hash(self) -> int:
    return boardmodule.hash(self._c_board)
  