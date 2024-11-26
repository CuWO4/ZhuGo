from agent.base import Agent

from agent.random_agent import RandomAgent
from go.goboard import Move, GameState
from go.gotypes import Player
from agent.helpers import is_point_an_eye
from utils.move_idx_transformer import idx_to_move, move_to_idx

import numpy as np
import math
import multiprocessing
import time

__all__ = [
  'MCTSAgent1Order'
]


class MCTSAgent1Order(Agent):
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
  def __init__(self, *, analysis_mode: bool = False, c: float = 1.0, n0: int = 500, beta: float = 0.3, max_second: int = 120, thread_n: int = 10,
               need_move_queue: bool = False, need_mcts_queue: bool = True):
    super().__init__(need_move_queue=need_move_queue, need_mcts_queue=need_mcts_queue)
    self.analysis_mode: bool = analysis_mode
    self.c: float = c
    self.n0: int = n0
    self.beta: float = beta
    self.random_agent: Agent = RandomAgent()
    self.max_second: int = max_second
    self.thread_n: int = thread_n

  def select_move(self, game_state: GameState) -> Move:
    assert game_state.board.num_rows == game_state.board.num_cols

    turn_start_timestamp = time.time()

    board = game_state.board

    row_n = board.num_rows
    col_n = board.num_cols
    policy_size = row_n * col_n + 2

    reward_sum = np.zeros(policy_size, dtype=np.float64)

    legal_mask = np.full(policy_size, -1e5, dtype=np.float64)
    for move in game_state.legal_moves():
      if move.is_pass:
        continue
      if not move.is_resign and \
        not is_point_an_eye(board, move.point, game_state.next_player):
        idx = move_to_idx(move, board.size)
        legal_mask[idx] = 0

    visited_times = np.zeros(policy_size, dtype=np.float64)

    with multiprocessing.Pool(self.thread_n) as pool:
      while self.analysis_mode or \
        ((np.sum(visited_times) < self.n0 or self.entropy(visited_times) > self.beta * math.log(policy_size)) and time.time() - turn_start_timestamp < self.max_second):
        ucb = self.calculate_ucb(reward_sum, visited_times, legal_mask)
        max_ucb_indexes = np.argwhere(ucb > np.max(ucb) - 0.02).flatten()
        indexes = np.random.choice(max_ucb_indexes, size=self.thread_n, replace=True)

        results = self.simulate_game(game_state, indexes, self.random_agent, pool, self.thread_n)

        for idx, winner in results:
          reward_sum[idx] += 1 if winner == game_state.next_player else 0
          visited_times[idx] += 1

        self.enqueue_mcts_data(reward_sum / (visited_times + 1e-8), visited_times, np.argmax(visited_times), board.size)

        print(f'{self.entropy(visited_times):.2f} {int(np.sum(visited_times))}')

        if self.analysis_mode:
          assert self.move_queue is not None
          human_move = self.dequeue_move(turn_start_timestamp, game_state)
          if human_move is not None:
            self.enqueue_empty_mcts_data(board.size)
            return human_move

    self.enqueue_empty_mcts_data(board.size)
    return idx_to_move(np.argmax(visited_times), board.size)

  def calculate_ucb(self, reward_sum: np.ndarray, visited_times: np.ndarray, legal_mask: np.ndarray):
    ucb = reward_sum / (visited_times + 1e-8) + self.c * np.sqrt(math.log(1 + np.sum(visited_times)) / (1 + visited_times))
    ucb += legal_mask
    return ucb

  @staticmethod
  def simulate_game(game_state: GameState, indexes: list[int], agent: Agent, pool: multiprocessing.Pool, thread_n: int) -> list[tuple[int, Player]]:
    results = pool.starmap(
      MCTSAgent1Order.simulate_worker,
      [(game_state, indexes[i], agent) for i in range(thread_n)]
    )
    return results

  @staticmethod
  def simulate_worker(game: GameState, idx: int, agent: Agent) -> tuple[int, Player]:
    game = game.apply_move(idx_to_move(idx, game.board.size))
    while not game.is_over():
      move = agent.select_move(game)
      game = game.apply_move(move)
    return (idx, game.winner())

  @staticmethod
  def entropy(array: np.ndarray) -> float:
    '''array do not need to be normalized'''
    distribution = (array + 1e-8) / np.sum(array + 1e-8)
    return - np.sum(distribution * np.log2(distribution))