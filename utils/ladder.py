from go.goboard import Board, Move, GoString
from go.gotypes import Point, Player

import copy

__all__ = [
  'LadderAnalysis',
  'analyze_ladder',
]

class LadderAnalysis:
  # public
  def is_stone_trapped_for_ladder(self, stone: Point) -> bool:
    return stone in self.__trapped_stones
  def is_stone_on_laddering_path(self, stone: Point) -> bool:
    return stone in self.__escape_paths
  def trapped_stones(self) -> set[Point]:
    return self.__trapped_stones
  def escape_paths(self) -> set[Point]:
    return self.__escape_paths

  # private
  def __init__(self) -> None:
    self.__trapped_stones: set[Point] = set()
    self.__escape_paths: set[Point] =set()

  # visible in module
  def _mark_stones_as_trapped(self, stones: set[Point]):
    self.__trapped_stones.update(stones)
  def _mark_escape_paths(self, stones: set[Point]):
    self.__escape_paths.update(stones)


def analyze_gostring(board: Board, point: Point) -> tuple[bool, int, set[Point]]:
  """Analyze a GoString to determine if it is captured in a ladder situation.

  Args:
    game_state (GameState): The current game state.
    point (Point): the point on gostring to be analyzed

  Returns:
    tuple: (bool, int, set). Indicates whether the string would be captured,
           the number of moves to escape/capture, and the escaping path.
  """

  current_string: GoString = board.get_go_string(point)

  if current_string is None: # captured
    return True, 0, set()

  if current_string.num_liberties > 1: # not captured
    return False, 0, set()

  escaping_player: Player = current_string.color
  chasing_player: Player = escaping_player.other

  escape_point: Point = next(iter(current_string.liberties))
  assert board.is_on_grid(escape_point)
  assert board.get(escape_point) is None

  if board.is_self_capture(escaping_player, escape_point):
    return True, 0, set() # captured

  after_escape_board = copy.deepcopy(board)
  after_escape_board.place_stone(escaping_player, escape_point)

  new_string = after_escape_board.get_go_string(escape_point)
  assert new_string is not None

  if new_string.num_liberties > 2:
    return False, 0, set()  # escaped

  chasing_candidates = [
    point for point in escape_point.neighbors()
    if board.is_on_grid(point)
       and board.get(point) is None
       and not board.is_self_capture(chasing_player, point)
  ]

  if chasing_candidates == []:
    return False, 0, set()

  is_captured = False
  escape_steps = 0
  escape_path = set()

  for chasing_point in chasing_candidates:
    after_chasing_board = copy.deepcopy(after_escape_board)
    after_chasing_board.place_stone(chasing_player, chasing_point)

    no_opponents_string_less_than_2_qi = True
    for neighbor in escape_point.neighbors():
      if not after_chasing_board.is_on_grid(neighbor):
        continue

      neighbor_string = after_chasing_board.get_go_string(neighbor)

      if neighbor_string is None or neighbor_string.color == escaping_player:
        continue

      if neighbor_string.num_liberties <= 1: # escaped
        no_opponents_string_less_than_2_qi = False
        break

    if not no_opponents_string_less_than_2_qi: # escaped
      continue

    sub_is_captured, sub_escape_steps, sub_escape_path = analyze_gostring(after_chasing_board, escape_point)

    if sub_is_captured: # captured
      is_captured = True
      escape_steps = max(escape_steps, sub_escape_steps + 1)
      escape_path |= sub_escape_path | set([escape_point])

  return is_captured, escape_steps, escape_path

def analyze_ladder(board: Board, threshold: int = 4) -> LadderAnalysis:
  """Analyze the board for ladder situations and return analysis.

  Args:
    board (Board): The current board.
    threshold (int): The minimum number of escape moves to consider.

  Returns:
    LadderAnalysis: The analysis results.
  """

  ladder_analysis = LadderAnalysis()
  visited_points: set[Point] = set()

  for point in [Point(row + 1, col + 1) for row in range(board.num_rows) for col in range(board.num_cols)]:
    if point in visited_points or board.get_go_string(point) is None:
      continue

    is_captured, escape_steps, escape_path = analyze_gostring(board, point)

    if is_captured and escape_steps > threshold:
      trapped_points = board.get_go_string(point).stones
      ladder_analysis._mark_stones_as_trapped(trapped_points)
      ladder_analysis._mark_escape_paths(escape_path)

    visited_points |= board.get_go_string(point).stones

  return ladder_analysis


# for test

import numpy as np
from utils.mcts_data import MCTSData
from utils.move_idx_transformer import move_to_idx

def ladder_analysis_to_mcts_data(ladder_analysis: LadderAnalysis, size: tuple[int, int]) -> MCTSData:
  q = np.zeros(size[0] * size[1])
  visited_times = np.zeros(size[0] * size[1])

  for point in [Point(row + 1, col + 1) for row in range(size[0]) for col in range(size[1])]:
    if ladder_analysis.is_stone_on_laddering_path(point):
      q[move_to_idx(Move.play(point), size)] = 0.5
      visited_times[move_to_idx(Move.play(point), size)] = 100
    if ladder_analysis.is_stone_trapped_for_ladder(point):
      q[move_to_idx(Move.play(point), size)] = 0
      visited_times[move_to_idx(Move.play(point), size)] = 100

  return MCTSData(q, visited_times, None, size)