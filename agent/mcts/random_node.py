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
    
    self.win_count: int = 0

    self._visited_times: np.ndarray = np.zeros(self.policy_size)
    self.margin_sums: np.ndarray = np.zeros(self.policy_size)
    
    self.total_visited_times: int = 0
    self.total_margin_sum: float = 0
    self.total_margin_biases: float = 0

    self.__is_q_dirty: bool = True
    self.__is_ucb_dirty: bool = True

    self.__q_cache: np.ndarray = np.zeros(self.policy_size, dtype=np.float64)
    self.__ucb_cache: np.ndarray = np.zeros(self.policy_size, dtype=np.float64)

  def propagate(self) -> list[GameResult]:
    if self.game_state.is_over():
      game_result = self.game_state.game_result()
      return [game_result]

    if self.depth == 0:
      indexes = exploring_move_indexes(self.ucb, self.pool._processes)
      results: list[tuple[int, GameResult]] = RandomNode.simulate_game(
        self.game_state,
        indexes,
        RandomAgent,
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
      if self.branches[move_idx] is None:
        self.branches[move_idx] = RandomNode(
          game_state=self.game_state.apply_move(move),
          pool=self.pool,
          c=self.c,
          depth=self.depth - 1,
        )

      game_results = self.branches[move_idx].propagate()

      for game_result in game_results:
        self.analyze_game_result(move_idx, game_result)

      return game_results

  def analyze_game_result(self, move_idx: int, game_result: GameResult):
    if game_result.winner == self.game_state.next_player:
      self.win_count += 1
    
    self.__is_q_dirty = True
    self.__is_ucb_dirty = True
    
    winning_margin = (1 if game_result.winner == self.game_state.next_player else -1) \
      * game_result.winning_margin

    self._visited_times[move_idx] += 1
    self.margin_sums[move_idx] += winning_margin
    
    self.total_margin_biases += (winning_margin - self.total_margin_sum / (self.total_visited_times + 1e-8)) ** 2
    self.total_visited_times += 1
    self.total_margin_sum += winning_margin
    
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

    if self.total_visited_times < 2 or self.total_margin_biases == 0:
      self.__q_cache.fill(0.5)
      self.__is_q_dirty = False
      return self.__q_cache

    mean = self.total_margin_sum / (self.total_visited_times + 1e-8)
    std = math.sqrt(self.total_margin_biases / (self.total_visited_times + 1e-8))

    self.__q_cache = (self.margin_sums / (self._visited_times + 1e-8) - mean) / std
    self.__q_cache = (self.__q_cache + 1) / 2
    
    self.__is_q_dirty = False
    return self.__q_cache
  
  @property
  def visited_times(self) -> np.ndarray[float]:
    return self._visited_times

  @property
  def win_rate(self) -> float:
    if self.total_visited_times == 0:
      return 0.5
    else:
      return self.win_count / self.total_visited_times

  @property
  def ucb(self) -> np.ndarray[float]:
    if not self.__is_ucb_dirty:
      return self.__ucb_cache

    self.__ucb_cache = self.q + self.c * np.sqrt(
      np.log(1 + np.sum(self._visited_times)) / (1 + self._visited_times))
    self.__ucb_cache += self.legal_mask
    return self.__ucb_cache

  @staticmethod
  def simulate_game(
    game_state: GameState,
    indexes: list[int],
    AgentType: type,
    pool: mp.Pool
  ) -> list[tuple[int, GameResult]]:
    results = pool.starmap(
      RandomNode.simulate_worker,
      [(i, game_state, indexes[i], AgentType) for i in range(pool._processes)]
    )
    return results

  @staticmethod
  def simulate_worker(thread_id: int, game, move_idx: int, AgentType: type) \
    -> tuple[int, GameResult]:
    usec = datetime.datetime.now().microsecond
    seed = thread_id + int(usec) & 0xFFFF_FFFF
    np.random.seed(seed)

    agent = AgentType()

    game = game.apply_move(idx_to_move(move_idx, game.board.size))
    while not game.is_over():
      move = agent.select_move(game)
      game = game.apply_move(move)

    game_result = game.game_result()

    return (move_idx, game_result)