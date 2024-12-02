import copy
from go.gotypes import Player, Point
import go.scoring as scoring
from go import zobrist

__all__ = [
  'Board',
  'GameState',
  'Move',
]

neighbor_tables = {}
corner_tables = {}


def init_neighbor_table(dim: int):
  rows, cols = dim
  new_table = {}
  for r in range(1, rows + 1):
    for c in range(1, cols + 1):
      p = Point(row=r, col=c)
      full_neighbors = p.neighbors()
      true_neighbors = [
        n for n in full_neighbors
        if 1 <= n.row <= rows and 1 <= n.col <= cols]
      new_table[p] = true_neighbors
  neighbor_tables[dim] = new_table


def init_corner_table(dim: int):
  rows, cols = dim
  new_table = {}
  for r in range(1, rows + 1):
    for c in range(1, cols + 1):
      p = Point(row=r, col=c)
      full_corners = [
        Point(row=p.row - 1, col=p.col - 1),
        Point(row=p.row - 1, col=p.col + 1),
        Point(row=p.row + 1, col=p.col - 1),
        Point(row=p.row + 1, col=p.col + 1),
      ]
      true_corners = [
        n for n in full_corners
        if 1 <= n.row <= rows and 1 <= n.col <= cols]
      new_table[p] = true_corners
  corner_tables[dim] = new_table


class IllegalMoveError(Exception):
  pass


class GoString():
  """Stones that are linked by a chain of connected stones of the
  same color.
  """
  def __init__(self, color, stones, liberties):
    self.color = color
    self.stones = frozenset(stones)
    self.liberties = frozenset(liberties)

  def without_liberty(self, point):
    new_liberties = self.liberties - set([point])
    return GoString(self.color, self.stones, new_liberties)

  def with_liberty(self, point):
    new_liberties = self.liberties | set([point])
    return GoString(self.color, self.stones, new_liberties)

  def merged_with(self, string):
    """Return a new string containing all stones in both strings."""
    assert string.color == self.color
    combined_stones = self.stones | string.stones
    return GoString(
      self.color,
      combined_stones,
      (self.liberties | string.liberties) - combined_stones)

  @property
  def num_liberties(self):
    return len(self.liberties)

  def __eq__(self, other):
    return isinstance(other, GoString) and \
      self.color == other.color and \
      self.stones == other.stones and \
      self.liberties == other.liberties

  def __deepcopy__(self, memodict={}):
    return GoString(self.color, self.stones, copy.deepcopy(self.liberties))


class Board():
  def __init__(self, num_rows: int, num_cols: int):
    self.num_rows: int = num_rows
    self.num_cols: int = num_cols
    self._grid = {}
    self._hash = zobrist.EMPTY_BOARD

    global neighbor_tables
    dim = (num_rows, num_cols)
    if dim not in neighbor_tables:
      init_neighbor_table(dim)
    if dim not in corner_tables:
      init_corner_table(dim)
    self.neighbor_table = neighbor_tables[dim]
    self.corner_table = corner_tables[dim]
    
  @property
  def size(self) -> tuple[int, int]:
    '''return row_n, col_n'''
    return self.num_rows, self.num_cols

  def neighbors(self, point: Point):
    return self.neighbor_table[point]

  def corners(self, point: Point):
    return self.corner_table[point]

  def place_stone(self, player: Player, point: Point):
    assert self.is_on_grid(point)
    if self._grid.get(point) is not None:
      print('Illegal play on %s' % str(point))
    assert self._grid.get(point) is None
    # 0. Examine the adjacent points.
    adjacent_same_color = []
    adjacent_opposite_color = []
    liberties = []
    for neighbor in self.neighbor_table[point]:
      neighbor_string = self._grid.get(neighbor)
      if neighbor_string is None:
        liberties.append(neighbor)
      elif neighbor_string.color == player:
        if neighbor_string not in adjacent_same_color:
          adjacent_same_color.append(neighbor_string)
      else:
        if neighbor_string not in adjacent_opposite_color:
          adjacent_opposite_color.append(neighbor_string)
    new_string = GoString(player, [point], liberties)
# tag::apply_zobrist[]
    # 1. Merge any adjacent strings of the same color.
    for same_color_string in adjacent_same_color:
      new_string = new_string.merged_with(same_color_string)
    for new_string_point in new_string.stones:
      self._grid[new_string_point] = new_string
    # Remove empty-point hash code.
    self._hash ^= zobrist.HASH_CODE[point, None]
    # Add filled point hash code.
    self._hash ^= zobrist.HASH_CODE[point, player]
# end::apply_zobrist[]

    # 2. Reduce liberties of any adjacent strings of the opposite
    #  color.
    # 3. If any opposite color strings now have zero liberties,
    #  remove them.
    for other_color_string in adjacent_opposite_color:
      replacement = other_color_string.without_liberty(point)
      if replacement.num_liberties:
        self._replace_string(other_color_string.without_liberty(point))
      else:
        self._remove_string(other_color_string)

  def _replace_string(self, new_string):
    for point in new_string.stones:
      self._grid[point] = new_string

  def _remove_string(self, string):
    for point in string.stones:
      # Removing a string can create liberties for other strings.
      for neighbor in self.neighbor_table[point]:
        neighbor_string = self._grid.get(neighbor)
        if neighbor_string is None:
          continue
        if neighbor_string is not string:
          self._replace_string(neighbor_string.with_liberty(point))
      self._grid[point] = None
      # Remove filled point hash code.
      self._hash ^= zobrist.HASH_CODE[point, string.color]
      # Add empty point hash code.
      self._hash ^= zobrist.HASH_CODE[point, None]

  def is_self_capture(self, player: Player, point: Point) -> bool:
    friendly_strings = []
    for neighbor in self.neighbor_table[point]:
      neighbor_string = self._grid.get(neighbor)
      if neighbor_string is None:
        # This point has a liberty. Can't be self capture.
        return False
      elif neighbor_string.color == player:
        # Gather for later analysis.
        friendly_strings.append(neighbor_string)
      else:
        if neighbor_string.num_liberties == 1:
          # This move is real capture, not a self capture.
          return False
    if all(neighbor.num_liberties == 1 for neighbor in friendly_strings):
      return True
    return False

  def will_capture(self, player: Player, point: Point) -> bool:
    for neighbor in self.neighbor_table[point]:
      neighbor_string = self._grid.get(neighbor)
      if neighbor_string is None:
        continue
      elif neighbor_string.color == player:
        continue
      else:
        if neighbor_string.num_liberties == 1:
          # This move would capture.
          return True
    return False

  def is_on_grid(self, point: Point) -> bool:
    return 1 <= point.row <= self.num_rows and \
      1 <= point.col <= self.num_cols

  def get(self, point: Point) -> Player | None:
    """Return the content of a point on the board.

    Returns None if the point is empty, or a Player if there is a
    stone on that point.
    """
    string = self._grid.get(point)
    if string is None:
      return None
    return string.color

  def get_go_string(self, point: Point) -> GoString:
    """Return the entire string of stones at a point.

    Returns None if the point is empty, or a GoString if there is
    a stone on that point.
    """
    string = self._grid.get(point)
    if string is None:
      return None
    return string

  def __str__(self) -> str:
    STONE_TO_CHAR = {
      None: '.',
      Player.white: 'o',
      Player.black: 'x'
    }
    str = ''
    for row in range(1, 1 + self.num_rows):
      for col in range(1, 1 + self.num_cols):
        str += STONE_TO_CHAR[self.get(Point(row=row, col=col))]
      str += '\n'
    return str

  def __eq__(self, other):
    return isinstance(other, Board) and \
      self.num_rows == other.num_rows and \
      self.num_cols == other.num_cols and \
      self._hash == other._hash

  def __deepcopy__(self, memodict={}):
    copied = Board(self.num_rows, self.num_cols)
    # Can do a shallow copy b/c the dictionary maps tuples
    # (immutable) to GoStrings (also immutable)
    copied._grid = copy.copy(self._grid)
    copied._hash = self._hash
    return copied

# tag::return_zobrist[]
  def zobrist_hash(self):
    return self._hash
# end::return_zobrist[]


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
    self.previous_state = previous_state
    self.komi: float = komi
    if previous_state is None:
      self.previous_states = frozenset()
    else:
      self.previous_states = frozenset(
        previous_state.previous_states |
        {(previous_state.next_player, previous_state.board.zobrist_hash())})
    self.last_move: Move = last_move

    self.is_privileged_mode: bool = is_privileged_mode

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

  def is_move_self_capture(self, player: Player, move: Move) -> bool:
    if not move.is_play:
      return False
    return self.board.is_self_capture(player, move.point)

  @property
  def situation(self) -> tuple[Player, Board]:
    return (self.next_player, self.board)

  def does_move_violate_ko(self, player: Player, move: Move) -> bool:
    if not move.is_play:
      return False
    if not self.board.will_capture(player, move.point):
      return False
    next_board = copy.deepcopy(self.board)
    next_board.place_stone(player, move.point)
    next_situation = (player.other, next_board.zobrist_hash())
    return next_situation in self.previous_states

  def is_valid_move(self, move: Move) -> bool:
    if self.is_over():
      return False
    if move.is_pass or move.is_resign:
      return True
    if move.is_undo:
      return self.is_privileged_mode
    return (
      self.board.get(move.point) is None and
      not self.is_move_self_capture(self.next_player, move) and
      not self.does_move_violate_ko(self.next_player, move))

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
    
    if other in self.previous_states:
      return True
    
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