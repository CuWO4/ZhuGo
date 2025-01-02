from .base import Agent
from .mcts.base import Node

from go.goboard import Move, GameState
from utils.mcts_data import MCTSData
from .mcts.utils import best_move_idx, cal_entropy
from utils.load_class_by_name import load_class_by_name

import numpy as np
import time
import multiprocessing as mp

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
  '''
  def __init__(self, *, node_type_name: str, node_settings: dict):
    super().__init__()
    self.NodeType: type = load_class_by_name(node_type_name)
    self.node_settings: dict = node_settings

    self.pool = mp.Pool(mp.cpu_count())
    
    self.root: Node | None = None

  def select_move(self, game_state: GameState) -> Move:
    turn_start_timestamp = time.time()

    board = game_state.board

    if self.root is None or not self.root.game_state.is_ancestor_of(game_state, 2):
      self.root = self.NodeType(game_state=game_state, pool=self.pool, **self.node_settings)
    else:
      for move in game_state - self.root.game_state:
        self.root = self.root.branch(move)

    while True:
      self.root.propagate()
      
      if self.ui is not None:
        self.ui.display_mcts(MCTSData(
          self.root.q,
          self.root.visited_times,
          best_move_idx(self.root.visited_times, self.root.q),
          self.root.win_rate,
          board.size
        ))

      self.print_statics_to_cmd(turn_start_timestamp)

      assert self.ui is not None
      human_move = self.ui.get_move(block=False)
      if human_move is not None:
        self.ui.display_mcts(MCTSData.empty(board.size))
        return human_move

  def print_statics_to_cmd(self, turn_start_timestamp: int):
    entropy = cal_entropy(self.root.visited_times)
    visited_times_sum = int(np.sum(self.root.visited_times))
    time_cost_sec = int(time.time() - turn_start_timestamp)
    time_cost_min = time_cost_sec // 60
    time_cost_sec %= 60
    time_cost_str = f'{time_cost_min:>2} min {time_cost_sec:>2} s'

    print('\033[?25l', end='', flush=False) # hide cursor
    print(f'{"entropy":>15}{"calculations":>15}{"time cost":>15}\n', end='', flush=False)
    print(f'{entropy:>15.2f}{visited_times_sum:>15d}{time_cost_str:>15}', end='', flush=True)
    print('\033[F\r', end='', flush=False) # set cursor position
