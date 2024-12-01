from .base import Node

from ..base import Agent
from .utils import exploring_move_indexes
from ..random_agent import RandomAgent

from go.goboard import GameState, Move
from go.scoring import GameResult
from utils.move_idx_transformer import idx_to_move, move_to_idx

import multiprocessing as mp
import numpy as np
import math
import datetime
from typing import TypeVar

__all__ = [
  'RandomNode'
]

T = TypeVar('T', bound='RandomNode')
class RandomNode(Node):
  def __init__(self, *, game_state: GameState, pool: mp.Pool, 
               c: float = 1.0, depth: int = 4):
    super().__init__(game_state=game_state, pool=pool, c=c)

    self.depth = depth

    self.move_winning_margins = [[] for _ in range(self.policy_size)]

    self.__is_q_dirty: bool = True
    self.__is_visited_times_dirty: bool = True
    self.__is_ucb_dirty: bool = True

    self.__q_cache: np.ndarray = np.array([0 for _ in range(self.policy_size)], dtype=np.float64)
    self.__visited_times_cache: np.ndarray = np.array([0 for _ in range(self.policy_size)], dtype=np.float64)
    self.__ucb_cache: np.ndarray = np.array([0 for _ in range(self.policy_size)], dtype=np.float64)

  def propagate(self) -> list[GameResult]:
    if self.game_state.is_over():
      game_result = self.game_state.game_result()
      return [game_result]

    if self.depth == 0:
      indexes = exploring_move_indexes(self.ucb, self.pool._processes)
      results: list[tuple[int, GameResult]] = RandomNode.simulate_game(
        self.game_state,
        indexes,
        RandomAgent(),
        self.pool
      )

      game_results = []

      for move_idx, game_result in results:
        self.analyze_game_result(move_idx, game_result)
        game_results.append(game_result)

      return game_results

    else:
      if self.legal_play_count > 0:
        move_idx = exploring_move_indexes(self.ucb, 1)[0]
      else:
        move_idx = move_to_idx(Move.pass_turn(), self.game_state.board.size)

      move = idx_to_move(move_idx, self.game_state.board.size)
      branch = self.branch(move)

      game_results = branch.propagate()

      for game_result in game_results:
        self.analyze_game_result(move_idx, game_result)

      return game_results

  def analyze_game_result(self, move_idx: int, game_result: GameResult):
    self.__is_q_dirty = True
    self.__is_ucb_dirty = True
    self.__is_visited_times_dirty = True
    self.move_winning_margins[move_idx].append(
      (1 if game_result.winner == self.game_state.next_player else -1) * game_result.winning_margin
    )

  def branch(self: T, move: Move) -> T:
    move_idx = move_to_idx(move, self.game_state.board.size)
    move = idx_to_move(move_idx, self.game_state.board.size)
    if self.branches[move_idx] is None:
      self.branches[move_idx] = RandomNode(
        game_state=self.game_state.apply_move(move),
        pool=self.pool,
        c=self.c,
        depth=self.depth,
      )

    self.branches[move_idx].depth = self.depth

    return self.branches[move_idx]


  @property
  def q(self) -> np.ndarray[float]:
    if not self.__is_q_dirty:
      return self.__q_cache

    def activate(arr: np.ndarray, scale: float = 2):
      arr = arr.clip(-scale + 1e-5, scale - 1e-5)
      arr = np.arcsin(arr / scale) / math.pi + 0.5
      return arr

    def sigmoid(arr: np.ndarray):
      return 1 / (1 + np.exp(-arr))

    all_margin = np.array(sum(self.move_winning_margins, start = []))

    if len(all_margin) < 2:
      self.__q_cache.fill(0.5)
      self.__is_q_dirty = False
      return self.__q_cache

    average = np.mean(all_margin)
    deviation = np.std(all_margin)
    
    if deviation == 0:
      self.__q_cache.fill(0.5)
      self.__is_q_dirty = False
      return self.__q_cache

    for idx, subarr in enumerate(self.move_winning_margins):
      if not subarr:
        self.__q_cache[idx] = 0.5
        continue

      subarr_np = np.array(subarr)
      subarr_np = (subarr_np - average) / deviation
      subarr_np = activate(subarr_np)
      self.__q_cache[idx] = np.mean(subarr_np)
    self.__q_cache = sigmoid(15 * (self.__q_cache - 0.5))

    self.__is_q_dirty = False
    return self.__q_cache

  @property
  def visited_times(self) -> np.ndarray[int]:
    if not self.__is_visited_times_dirty:
      return self.__visited_times_cache

    for idx, subarr in enumerate(self.move_winning_margins):
      self.__visited_times_cache[idx] = len(subarr)
    self.__is_visited_times_dirty = False
    return self.__visited_times_cache

  @property
  def ucb(self) -> np.ndarray[float]:
    if not self.__is_ucb_dirty:
      return self.__ucb_cache

    self.__ucb_cache = self.q + self.c * np.sqrt(
      np.log(1 + np.sum(self.visited_times)) / (1 + self.visited_times))
    self.__ucb_cache += self.legal_mask
    return self.__ucb_cache

  @staticmethod
  def simulate_game(
    game_state: GameState,
    indexes: list[int],
    agent: Agent,
    pool: mp.Pool
  ) -> list[tuple[int, GameResult]]:
    results = pool.starmap(
      RandomNode.simulate_worker,
      [(i, game_state, indexes[i], agent) for i in range(pool._processes)]
    )
    return results

  @staticmethod
  def simulate_worker(thread_id: int, game: GameState, move_idx: int, agent: Agent) \
    -> tuple[int, GameResult]:
    usec = datetime.datetime.now().microsecond
    seed = thread_id + int(usec) & 0xFFFF_FFFF
    np.random.seed(seed)

    game = game.apply_move(idx_to_move(move_idx, game.board.size))
    while not game.is_over():
      move = agent.select_move(game)
      game = game.apply_move(move)

    game_result = game.game_result()

    return (move_idx, game_result)