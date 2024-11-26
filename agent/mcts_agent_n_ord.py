from agent.base import Agent

from agent.random_agent import RandomAgent
from go.goboard import Move, GameState
from go.gotypes import Player
from agent.helpers import is_point_an_eye
from utils.move_idx_transformer import idx_to_move, move_to_idx

import numpy as np
import multiprocessing
import math
import time

__all__ = [
  'MCTSAgentNOrder'
]

class MCTSAgentNOrder(Agent):
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
               max_depth: int = 2, c: float = 1.0, n0: int = 500, beta: float = 0.3, max_second: int = 120, thread_n: int = 10,
               need_move_queue: bool = False, need_mcts_queue: bool = True):
    super().__init__(need_move_queue=need_move_queue, need_mcts_queue=need_mcts_queue)
    self.analysis_mode: bool = analysis_mode
    self.max_depth = max_depth
    self.c: float = c
    self.n0: int = n0
    self.beta: float = beta
    self.random_agent: Agent = RandomAgent()
    self.max_second: int = max_second
    self.thread_n: int = thread_n
    
  def select_move(self, game_state: GameState) -> Move:
    turn_start_timestamp = time.time()

    board = game_state.board
    
    root = MCTSNode(self.max_depth - 1, self.thread_n, game_state, c=self.c)
    
    while self.analysis_mode or \
      ((np.sum(root.visited_times) < self.n0 
        or self.entropy(root.visited_times) > self.beta * math.log(root.policy_size)) 
        and time.time() - turn_start_timestamp < self.max_second):

      # multiprocessing need to pickle serialize the object, which is extremely slow
      root.propagate()
      
      self.enqueue_mcts_data(
        root.reward_sum / (root.visited_times + 1e-8), 
        root.visited_times, 
        np.argmax(root.visited_times), board.size
      )

      print(f'{self.entropy(root.visited_times):.2f} {int(np.sum(root.visited_times))}')

      if self.analysis_mode:
        assert self.move_queue is not None
        human_move = self.dequeue_move(turn_start_timestamp, game_state)
        if human_move is not None:
          self.enqueue_empty_mcts_data(board.size)
          return human_move
          
    self.enqueue_empty_mcts_data(board.size)
    return idx_to_move(np.argmax(root.visited_times), board.size)

  @staticmethod
  def entropy(array: np.ndarray) -> float:
    '''array do not need to be normalized'''
    distribution = (array + 1e-8) / np.sum(array + 1e-8)
    return - np.sum(distribution * np.log2(distribution))
  
class MCTSNode:
  def __init__(self, depth: int, thread_n: int, game_state: GameState, *, c: float):
    self.game_state: GameState = game_state

    self.depth = depth
    self.thread_n = thread_n
    
    self.c: float = c
    
    self.row_n: int = game_state.board.num_rows
    self.col_n: int = game_state.board.num_cols
    self.policy_size: int = self.row_n * self.col_n + 2
    
    self.branches: list[MCTSNode] = [None] * self.policy_size

    self.reward_sum = np.zeros(self.policy_size, dtype=np.float64)

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
        
    self.visited_times = np.zeros(self.policy_size, dtype=np.float64)

  def propagate(self) -> dict[Player, int]:
    if self.game_state.is_over():
      winner = self.game_state.winner()
      return { winner: 1, winner.other: 0 }
    
    if self.depth == 0:
      ucb = self.calculate_ucb()
      max_ucb_indexes = np.argwhere(ucb > np.max(ucb) - 0.02).flatten()
      indexes = np.random.choice(max_ucb_indexes, size=self.thread_n, replace=True)
      with multiprocessing.Pool(self.thread_n) as pool:
        results = self.simulate_game(self.game_state, indexes, RandomAgent(), pool, self.thread_n)
        
      winning_dict = {
        Player.black: 0,
        Player.white: 0
      }
      
      for move_idx, winner in results:
        self.reward_sum[move_idx] += 1 if winner == self.game_state.next_player else 0
        self.visited_times[move_idx] += 1
        winning_dict[winner] += 1
        
      return winning_dict
      
    else:
      if self.legal_move_count > 0:
        ucb = self.calculate_ucb()
        max_ucb_indexes = np.argwhere(ucb == np.max(ucb)).flatten()
        move_idx = np.random.choice(max_ucb_indexes)
      else:
        move_idx = move_to_idx(Move.pass_turn(), self.game_state.board.size)
      
      if not self.branches[move_idx]:
        move = idx_to_move(move_idx, self.game_state.board.size)
        self.branches[move_idx] = MCTSNode(self.depth - 1, self.thread_n, self.game_state.apply_move(move), c=self.c)
        
      winning_dict = self.branches[move_idx].propagate()

      for winner, win_games in winning_dict.items():
        if winner == self.game_state.next_player:
          self.reward_sum[move_idx] += win_games
        self.visited_times[move_idx] += win_games
      
      return winning_dict
    
  @staticmethod
  def simulate_game(game_state: GameState, 
                    indexes: list[int], agent: Agent, 
                    pool: multiprocessing.Pool, thread_n: int) -> list[tuple[int, Player]]:
    results = pool.starmap(
      MCTSNode.simulate_worker,
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
    
  def calculate_ucb(self) -> float:
    ucb = self.reward_sum / (self.visited_times + 1e-8) + self.c * np.sqrt(math.log(1 + np.sum(self.visited_times)) / (1 + self.visited_times))
    ucb += self.legal_mask
    return ucb