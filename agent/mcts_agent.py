from .base import Agent
from .mcts.base import Node

from go.goboard import Move, GameState
from go.gotypes import Player
from utils.mcts_data import MCTSData
from .mcts.utils import best_move_idx, cal_entropy
from utils.load_class_by_name import load_class_by_name
from .mcts.monitor import Monitor

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
    
    self.data_connection, data_connection = mp.Pipe()
    self.monitor = mp.Process(target = Monitor, args = (data_connection,))
    self.monitor.start()

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

      self.update_monitor(game_state, turn_start_timestamp)

      assert self.ui is not None
      while True:
        human_move = self.ui.get_move(block=False)
        if human_move is None:
          break
        if game_state.is_valid_move(human_move):
          self.ui.display_mcts(MCTSData.empty(board.size))
          return human_move

  def update_monitor(self, game_state: GameState, turn_start_timestamp: int):
    entropy = cal_entropy(self.root.visited_times)
    visited_times_sum = int(np.sum(self.root.visited_times))
    time_cost_sec = int(time.time() - turn_start_timestamp)

    self.data_connection.send((
      game_state.turn,
      self.root.win_rate if game_state.next_player == Player.black else 1 - self.root.win_rate,
      entropy,
      visited_times_sum,
      time_cost_sec,
    ))
