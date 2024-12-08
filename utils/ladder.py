from go.goboard import Board, Move
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

  Note:
    board will be modified
  """
  assert board.in_board(point)

  assert board.get(point) != None

  if board.qi(point) > 1: # not captured
    return False, 0, set()

  escaping_player: Player = board.get(point)
  chasing_player: Player = escaping_player.other

  escape_point: Point = board.get_random_qi_pos(point)
  assert board.in_board(escape_point)
  assert board.get(escape_point) is None

  if not board.is_valid_move(escape_point, escaping_player):
    return True, 0, set() # captured

  after_escape_board = board
  after_escape_board.place_stone(escaping_player, escape_point)

  if after_escape_board.qi(escape_point) > 2:
    return False, 0, set()  # escaped
  
  if after_escape_board.qi(escape_point) <= 1:
    return True, 1, set([escape_point])

  chasing_candidates = [
    point for point in escape_point.neighbors()
    if after_escape_board.in_board(point)
       and after_escape_board.is_valid_move(point, chasing_player)
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
      if not after_chasing_board.in_board(neighbor):
        continue

      if after_chasing_board.get(neighbor) is None \
        or after_chasing_board.get(neighbor) == escaping_player:
        continue

      if after_chasing_board.qi(neighbor) <= 1: # escaped
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

# For complex situations, each analysis takes about 5ms, and multiprocessing has not been able yet. 
# It is not a performance hotspot for the time being, so more complex optimizations are not considered.
def analyze_ladder(board: Board, threshold: int = 4) -> LadderAnalysis:
  """Analyze the board for ladder situations and return analysis.

  Args:
    board (Board): The current board.
    threshold (int): The minimum number of escape moves to consider.

  Returns:
    LadderAnalysis: The analysis results.
  """
  
  def get_go_string(board: Board, point: Point, visited_points: set[Point]) -> set[Point]:
    if point in visited_points:
      return set()
    visited_points.update([point])
    
    stone = board.get(point)
    if stone is None:
      return set()
    
    stones = set([point])
    
    for neighbor in (Point(row = point.row + dx, col = point.col + dy) for dx in (-1, 1) for dy in (-1, 1)):
      if not board.in_board(neighbor):
        continue

      if board.get(neighbor) == stone:
        stones |= get_go_string(board, neighbor, visited_points)
      
    return stones

  ladder_analysis = LadderAnalysis()
  visited_points: set[Point] = set()

  for point in [Point(row + 1, col + 1) for row in range(board.num_rows) for col in range(board.num_cols)]:
    if point in visited_points or board.get(point) is None:
      continue

    go_string = get_go_string(board, point, set())

    is_captured, escape_steps, escape_path = analyze_gostring(copy.deepcopy(board), point)

    if is_captured and escape_steps > threshold:
      ladder_analysis._mark_stones_as_trapped(go_string)
      ladder_analysis._mark_escape_paths(escape_path)

    visited_points |= go_string

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

  return MCTSData(q, visited_times, None, None, size)