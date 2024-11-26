from agent.base import Agent

from agent.random_agent import RandomAgent
from go.goboard import Move, GameState
from go.gotypes import Player
from agent.helpers import is_point_an_eye
from utils.move_idx_transformer import idx_to_move, move_to_idx
from ui.utils import MCTSData

import numpy as np
import math
import copy
import multiprocessing
import time

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
  def __init__(self, c: float = 0.5, n0: int = 500, beta: float = 0.3, max_second: int = 120, thread_n: int = 10):
    super().__init__()
    self.c: float = c
    self.n0: int = n0
    self.beta: float = beta
    self.random_agent: Agent = RandomAgent()
    self.max_second: int = max_second
    self.thread_n: int = thread_n
    
    self.mcts_queue: multiprocessing.Queue | None = None

  def subscribe_mcts_queue(self, mcts_queue: multiprocessing.Queue):
    self.mcts_queue = mcts_queue

  def select_move(self, game_state: GameState) -> Move:
    assert game_state.board.num_rows == game_state.board.num_cols

    board = game_state.board

    row_n = board.num_rows
    col_n = board.num_cols
    policy_size = row_n * col_n + 2

    reward_sum = np.zeros(policy_size, dtype=np.float64)

    legal_mask = np.zeros(policy_size, dtype=np.float64)
    for move in game_state.legal_moves():
      if not move.is_pass and \
        not move.is_resign and \
        not is_point_an_eye(board, move.point, game_state.next_player):
        idx = move_to_idx(move, board.size())
        legal_mask[idx] = 1

    visited_times = np.zeros(policy_size, dtype=np.float64)

    with multiprocessing.Pool(self.thread_n) as pool:
      starttime = time.time()
      while np.sum(visited_times) < self.n0 or self.entropy(visited_times) > self.beta * math.log2(policy_size):
        ucb = self.calculate_ucb(reward_sum, visited_times, legal_mask)        
        max_ucb_indexes = np.argwhere(ucb == np.max(ucb)).flatten()
        indexes = np.random.choice(max_ucb_indexes, size=self.thread_n, replace=True)

        results = self.simulate_game(game_state, indexes, self.random_agent, pool, self.thread_n)

        for idx, winner in results:
          reward_sum[idx] += 1 if winner == game_state.next_player else 0
          visited_times[idx] += 1
          
        self.update_mcts_gui(reward_sum, visited_times, board.size())
          
        print(f'{self.entropy(visited_times):.2f} {int(np.sum(visited_times))}')
        
        if time.time() - starttime > self.max_second:
          break

    self.clean_mcts_gui((row_n, col_n))
    return idx_to_move(np.argmax(visited_times), board.size())
  
  def calculate_ucb(self, reward_sum: np.ndarray, visited_times: np.ndarray, legal_mask: np.ndarray):
    ucb = reward_sum / (visited_times + 1e-8) + self.c * np.sqrt(math.log(1 + np.sum(visited_times)) / (1 + visited_times))
    ucb *= legal_mask
    return ucb
  
  def update_mcts_gui(self, reward_sum: np.ndarray, visited_times: np.ndarray, size: tuple[int, int]):
    if self.mcts_queue is None:
      return
    q = reward_sum / (visited_times + 1e-8)
    self.mcts_queue.put(MCTSData(q, visited_times, np.argmax(visited_times), size))

  def clean_mcts_gui(self, size: tuple[int, int]):
    if self.mcts_queue is None:
      return
    self.mcts_queue.put(MCTSData.empty(size))
    
  @staticmethod
  def simulate_game(game_state: GameState, indexes: list[int], agent: Agent, pool: multiprocessing.Pool, thread_n: int) -> list[tuple[int, Player]]:
    results = pool.starmap(
      MCTSAgent.simulate_worker, 
      [(GameState(copy.deepcopy(game_state.board), game_state.next_player, None, None), indexes[i], agent) for i in range(thread_n)]
    )
    return results
  
  @staticmethod
  def simulate_worker(game: GameState, idx: int, agent: Agent) -> tuple[int, Player]:
    game.apply_move(idx_to_move(idx, game.board.size()))
    while not game.is_over():
      move = agent.select_move(game)
      game = game.apply_move(move)
    return (idx, game.winner())
  
  @staticmethod
  def entropy(array: np.ndarray) -> float:
    '''array do not need to be normalized'''
    distribution = (array + 1e-8) / np.sum(array + 1e-8)
    return - np.sum(distribution * np.log2(distribution))