import copy
from go.gotypes import Player, Point
import go.scoring as scoring
from cboard.cboard import cBoard, does_violate_ko

__all__ = [
  'Board',
  'GameState',
  'Move',
]


class IllegalMoveError(Exception):
  pass


class Board():
  def __init__(self, num_rows: int, num_cols: int, c_board: cBoard | None = None):
    self.num_rows: int = num_rows
    self.num_cols: int = num_cols
    if c_board is None:
      self.c_board: cBoard = cBoard(num_rows, num_cols)
    else:
      self.c_board = c_board
      
  @staticmethod
  def c_player_to_py_player(c_player: int) -> Player:
    if c_player == 0:
      return None
    elif c_player == 1:
      return Player.black
    elif c_player == 2:
      return Player.white
    else:
      raise ValueError(f'unknown c player {c_player}')
    
  @staticmethod
  def py_player_to_c_player(py_player: Player) -> int:
    if py_player is None:
      return 0
    elif py_player == Player.black:
      return 1
    elif py_player == Player.white:
      return 2
    else:
      raise ValueError(f'unknown py player {py_player}')

  @property
  def size(self) -> tuple[int, int]:
    '''return row_n, col_n'''
    return self.num_rows, self.num_cols
  
  def get(self, point: Point) -> Player | None:
    c_player = self.c_board.get(point.row - 1, point.col - 1)
    return self.c_player_to_py_player(c_player)
  
  def qi(self, point: Point) -> int:
    return self.c_board.qi(point.row - 1, point.col - 1)
  
  def get_random_qi_pos(self, point: Point) -> Point:
    c_row, c_col = self.c_board.get_random_qi_pos(point.row - 1, point.col - 1)
    return Point(row = c_row + 1, col = c_col + 1)
  
  def is_valid_move(self, point: Point, player: Player) -> bool:
    c_player = self.py_player_to_c_player(player)
    return self.c_board.is_valid_move(c_player, point.row - 1, point.col - 1)

  def place_stone(self, player: Player, point: Point):
    c_player = self.py_player_to_c_player(player)
    self.c_board.place_stone(c_player, point.row - 1, point.col - 1)

  def in_board(self, point: Point) -> bool:
    return 1 <= point.row <= self.num_rows and \
      1 <= point.col <= self.num_cols

  def __str__(self) -> str:
    STONE_TO_CHAR = {
      None: '.',
      Player.white: 'o',
      Player.black: 'x'
    }
    str = '  '
    for col in range(self.num_cols):
      str += ' ' + chr(ord('A') + col)
    str += '\n'
    for row in range(1, 1 + self.num_rows):
      str += f'{row:>2d}'
      for col in range(1, 1 + self.num_cols):
        str += ' ' + STONE_TO_CHAR[self.get(Point(row=row, col=col))]
      str += '\n'
    return str

  def __eq__(self, other):
    return isinstance(other, Board) \
      and self.num_rows == other.num_rows \
      and self.num_cols == other.num_cols \
      and self.hash() == other.hash()

  def __deepcopy__(self, memo={}):
    return Board(self.num_rows, self.num_cols, copy.deepcopy(self.c_board))

  def hash(self) -> int:
    return self.c_board.hash()


class Move():
  """Any action a player can play on a turn.

  Exactly one of is_play, is_pass, is_resign will be set.
  
  Undo will only be permitted when the game applied with undo move is in privileged mode.
  """
  def __init__(
    self, 
    point: Point = None, 
    is_pass: bool = False, 
    is_resign: bool = False, 
    is_undo: bool = False
  ):
    assert (point is not None) + is_pass + is_resign + is_undo == 1
    self.point: Point = point
    self.is_play: bool = (self.point is not None)
    self.is_pass: bool = is_pass
    self.is_resign: bool = is_resign
    self.is_undo: bool = is_undo

  @staticmethod
  def play(point: Point):
    """A move that places a stone on the board."""
    return Move(point=point)

  @staticmethod
  def pass_turn():
    return Move(is_pass=True)

  @staticmethod
  def resign():
    return Move(is_resign=True)

  @staticmethod
  def undo():
    return Move(is_undo=True)

  def __str__(self):
    if self.is_pass:
      return 'pass'
    elif self.is_resign:
      return 'resign'
    elif self.is_undo:
      return 'undo'
    elif self.is_play:
      point = self.point
      return f'(r {point.row}, c {point.col})'
    assert False

  def __hash__(self):
    return hash((
      self.is_play,
      self.is_pass,
      self.is_resign,
      self.is_undo,
      self.point
    ))

  def  __eq__(self, other):
    return (
      self.is_play,
      self.is_pass,
      self.is_resign,
      self.is_undo,
      self.point
    ) == (
      other.is_play,
      other.is_pass,
      other.is_resign,
      other.is_undo,
      other.point
    )


class GameState():
  def __init__(self, board: Board, next_player: Player, previous_state, last_move: Move, komi: float,
               *, is_privileged_mode: bool = False):
    '''only privileged mode is on, undo move is allowed
    '''
    self.board: Board = board
    self.next_player: Player = next_player
    self.previous_state: GameState = previous_state
    self.komi: float = komi
    self.last_move: Move = last_move

    self.is_privileged_mode: bool = is_privileged_mode
    
  # ignore previous states which is only useful for undo move
  def __getstate__(self) -> dict:
    state = self.__dict__.copy()
    if state['previous_state'] is not None:
      state['previous_state'].previous_state = None
    return state
    
  def apply_move(self, move: Move):
    """Return the new GameState after applying the move."""
    assert not self.is_over()
    
    if move.is_undo:
      assert self.is_privileged_mode
      if self.previous_state is not None:
        return self.previous_state 
      else:
        return self
    
    next_board = copy.deepcopy(self.board)
    if move.is_play:
      next_board.place_stone(self.next_player, move.point)
      
    return GameState(
      next_board, 
      self.next_player.other, 
      self, 
      move, 
      self.komi, 
      is_privileged_mode = self.is_privileged_mode
    )

  @staticmethod
  def new_game(board_size: tuple[int] = (19, 19), komi: float = 7.5, *, is_privileged_mode: bool = False):
    return GameState(Board(*board_size), Player.black, None, None, komi, is_privileged_mode = is_privileged_mode)

  def does_move_violate_ko(self, player: Player, move: Move) -> bool:
    if not move.is_play:
      return False
    if self.previous_state is None:
      return False
    c_board = self.board.c_board
    c_player = Board.py_player_to_c_player(player)
    c_row, c_col = move.point.row - 1, move.point.col - 1
    c_last_board = self.previous_state.board.c_board
    return does_violate_ko(c_board, c_player, c_row, c_col, c_last_board)

  def is_valid_move(self, move: Move) -> bool:
    if self.is_over():
      return False
    if move.is_pass or move.is_resign:
      return True
    if move.is_undo:
      return self.is_privileged_mode
    return self.board.get(move.point) is None \
      and self.board.is_valid_move(move.point, self.next_player) \
      and not self.does_move_violate_ko(self.next_player, move)

  def is_over(self) -> bool:
    if self.last_move is None:
      return False
    if self.last_move.is_resign:
      return True 
    second_last_move = self.previous_state.last_move
    if second_last_move is None:
      return False
    return self.last_move.is_pass and second_last_move.is_pass

  def legal_moves(self) -> list[Move]:
    '''not include undo move'''
    moves = []
    for row in range(1, self.board.num_rows + 1):
      for col in range(1, self.board.num_cols + 1):
        move = Move.play(Point(row, col))
        if self.is_valid_move(move):
          moves.append(move)
    # These two moves are always legal.
    moves.append(Move.pass_turn())
    moves.append(Move.resign())

    return moves
  
  def game_result(self) -> scoring.GameResult:
    return scoring.compute_game_result(self)

  def winner(self) -> Player:
    if not self.is_over():
      return None
    if self.last_move.is_resign:
      return self.next_player
    return self.game_result().winner
  
  def is_ancestor_of(self, other, max_n: int) -> bool:
    if not isinstance(other, GameState):
      return False
    
    state: GameState = other
    for _ in range(max_n + 1):
      if state is None:
        return False
      if state.board == self.board:
        return True
      state = state.previous_state

    return False
  
  def __sub__(self, other) -> list[Move]:
    assert isinstance(other, GameState)
    
    move_list = []
    state = self
    while state != other and state.board != other.board:
      move_list = [state.last_move] + move_list
      state = state.previous_state
      assert state is not None # triggered if other is not self's ancestor

    return move_list