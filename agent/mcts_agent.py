from agent.base import Agent

from agent.random_agent import RandomAgent
from go.goboard import Move, GameState
from go.gotypes import Player
from agent.helpers import is_point_an_eye
from utils.move_idx_transformer import idx_to_move, move_to_idx
from go.scoring import GameResult

import numpy as np
import multiprocessing
import math
import time
import datetime

__all__ = [
  'MCTSAgent'
]

class MCTSAgent(Agent):
  '''
  Parameters
  ------------------
  c
    a larger c makes the agent tend to explore uncertain branches,
    while a smaller c makes the agent tend to increase the confidence of
    the dominant branch.
  n0
    least simulate plays
  beta
    ratio of least determine entropy and entropy of even distribution on policy space
  max second
    max time (second) for each move
  thread_n
    simulate thread counts
  '''
  def __init__(self, *, analysis_mode: bool = False,
               max_depth: int = 8, c: float = 1.0, n0: int = 500, beta: float = 0.3, 
               max_second: int = 120, thread_n: int = 10,
               need_move_queue: bool = False, need_mcts_queue: bool = True):
    super().__init__(need_move_queue=need_move_queue, need_mcts_queue=need_mcts_queue)
    self.analysis_mode: bool = analysis_mode
    self.max_depth = max_depth
    self.c: float = c
    self.n0: int = n0
    self.beta: float = beta
    self.max_second: int = max_second
    self.thread_n: int = thread_n
    self.pool = multiprocessing.Pool(thread_n)

    self.root: MCTSNode | None = None

  def select_move(self, game_state: GameState) -> Move:
    turn_start_timestamp = time.time()

    board = game_state.board

    if self.root is None or not self.root.game_state.is_ancestor_of(game_state, 2):
      self.root = MCTSNode(self.max_depth, self.thread_n, self.pool, game_state, c=self.c)
    else:
      self.switch_branch(game_state - self.root.game_state)

    while self.analysis_mode or \
      ((np.sum(self.root.visited_times) < self.n0
        or self.entropy(self.root.visited_times) > self.beta * math.log(self.root.policy_size))
        and time.time() - turn_start_timestamp < self.max_second):

      # multiprocessing need to pickle serialize the object, which is extremely slow
      self.root.propagate()

      self.enqueue_mcts_data(
        self.root.q,
        self.root.visited_times,
        MCTSAgent.best_move_idx(self.root.visited_times, self.root.q),
        board.size
      )

      self.print_statics_to_cmd(turn_start_timestamp)

      if self.analysis_mode:
        assert self.move_queue is not None
        human_move = self.dequeue_move(turn_start_timestamp, game_state)
        if human_move is not None:
          self.enqueue_empty_mcts_data(board.size)
          return human_move

    self.enqueue_empty_mcts_data(board.size)
    return MCTSAgent.best_move_idx(self.root.visited_times, self.root.q)

  def print_statics_to_cmd(self, turn_start_timestamp: int):
    entropy = MCTSAgent.entropy(self.root.visited_times)
    visited_times_sum = int(np.sum(self.root.visited_times))
    time_cost_sec = int(time.time() - turn_start_timestamp)
    time_cost_min = time_cost_sec // 60
    time_cost_sec %= 60
    time_cost_str = f'{time_cost_min:>2} min {time_cost_sec:>2} s'

    print('\033[?25l', end='', flush=False) # hide cursor
    print(f'{"entropy":>15}{"calculations":>15}{"time cost":>15}\n', end='', flush=False)
    print(f'{entropy:>15.2f}{visited_times_sum:>15d}{time_cost_str:>15}', end='', flush=True)
    print('\033[F\r', end='', flush=False) # set cursor position


  def switch_branch(self, move_list: list[Move]):
    assert self.root is not None

    node: MCTSNode = self.root
    for move in move_list:
      move_idx = move_to_idx(move, node.game_state.board.size)
      if node.branches[move_idx] is None:
        node.branches[move_idx] = MCTSNode(
          self.max_depth,
          self.thread_n,
          self.pool,
          node.game_state.apply_move(move),
          c=self.c
        )
      node = node.branches[move_idx]
      node.depth = self.max_depth

    self.root = node

  @staticmethod
  def exploring_move_indexes(ucb: list[float], size: int) -> list[int]:
    max_ucb_indexes = np.argwhere(ucb == np.max(ucb)).flatten()
    move_indexes = np.random.choice(max_ucb_indexes, size=size)
    return move_indexes

  @staticmethod
  def best_move_idx(
    visited_times: np.ndarray,
    q: np.ndarray
  ) -> int:
    max_visited_indexes = np.argwhere(visited_times == np.max(visited_times)).flatten()
    max_q_idx_in_max_visited = np.argmax(q[max_visited_indexes])
    max_idx = max_visited_indexes[max_q_idx_in_max_visited]
    return max_idx

  @staticmethod
  def entropy(array: np.ndarray) -> float:
    '''array do not need to be normalized'''
    distribution = (array + 1e-8) / np.sum(array + 1e-8)
    return - np.sum(distribution * np.log2(distribution))

class MCTSNode:
  def __init__(self, depth: int, thread_n: int, pool: multiprocessing.Pool, game_state: GameState, 
               *, c: float):
    self.game_state: GameState = game_state

    self.depth = depth
    self.thread_n = thread_n
    self.pool = pool

    self.c: float = c

    self.row_n: int = game_state.board.num_rows
    self.col_n: int = game_state.board.num_cols
    self.policy_size: int = self.row_n * self.col_n + 2

    self.branches: list[MCTSNode] = [None] * self.policy_size

    self.move_winning_margins: list[list[float]] = [[] for _ in range(self.policy_size)]

    self.legal_mask = np.full(self.policy_size, -1e5, dtype=np.float64)
    self.legal_move_count = 0
    for move in game_state.legal_moves():
      if move.is_pass:
        continue
      if not move.is_resign and \
        not is_point_an_eye(self.game_state.board, move.point, game_state.next_player):
        idx = move_to_idx(move, self.game_state.board.size)
        self.legal_mask[idx] = 0
        self.legal_move_count += 1

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
      indexes = MCTSAgent.exploring_move_indexes(self.ucb, self.thread_n)
      results: list[tuple[int, GameResult]] = MCTSNode.simulate_game(
        self.game_state, 
        indexes, 
        RandomAgent(), 
        self.pool, 
        self.thread_n
      )

      game_results = []

      for move_idx, game_result in results:
        self.analyze_game_result(move_idx, game_result)
        game_results.append(game_result)

      return game_results

    else:
      if self.legal_move_count > 0:
        move_idx = MCTSAgent.exploring_move_indexes(self.ucb, 1)[0]
      else:
        move_idx = move_to_idx(Move.pass_turn(), self.game_state.board.size)
        
      if not self.branches[move_idx]:
        move = idx_to_move(move_idx, self.game_state.board.size)
        self.branches[move_idx] = MCTSNode(
          self.depth - 1, 
          self.thread_n, 
          self.pool, 
          self.game_state.apply_move(move), 
          c=self.c
        )

      game_results = self.branches[move_idx].propagate()

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

  @property
  def q(self) -> np.ndarray:
    if not self.__is_q_dirty:
      return self.__q_cache
    
    def activate(arr: np.ndarray, scale: float = 2):
      arr = arr.clip(-scale, scale)
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
  def visited_times(self) -> np.ndarray:
    if not self.__is_visited_times_dirty:
      return self.__visited_times_cache
    
    for idx, subarr in enumerate(self.move_winning_margins):
      self.__visited_times_cache[idx] = len(subarr) / self.thread_n
    self.__is_visited_times_dirty = False
    return self.__visited_times_cache

  @property
  def ucb(self) -> np.ndarray:
    if not self.__is_ucb_dirty:
      return self.__ucb_cache

    self.__ucb_cache = self.q + self.c * np.sqrt(
      np.log(1 + np.sum(self.visited_times)) / (1 + self.visited_times))
    self.__ucb_cache += self.legal_mask
    return self.__ucb_cache
  
  @staticmethod
  def simulate_game(
    game_state: GameState,
    indexes: list[int], agent: Agent,
    pool: multiprocessing.Pool, thread_n: int
  ) -> list[tuple[int, GameResult]]:
    results = pool.starmap(
      MCTSNode.simulate_worker,
      [(i, game_state, indexes[i], agent) for i in range(thread_n)]
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