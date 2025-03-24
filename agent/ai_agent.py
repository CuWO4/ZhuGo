from .mcts_agent import MCTSAgent

from ai.zhugo import ZhuGo
from ai.encoder.zhugo_encoder import ZhuGoEncoder
from ai.manager import load
from .mcts.ai_node import AINode
from go.goboard import GameState, Move
from utils.mcts_data import MCTSData
from .mcts.utils import best_move_idx

import torch
import time

__all__ = [
  'AIAgent'
]


class AIAgent(MCTSAgent):
  def __init__(
    self, 
    model: str | ZhuGo, noise_intensity: float = 0, 
    *, node_settings: dict
  ):
    super().__init__(node_type_name='agent.mcts.ai_node.AINode', node_settings=node_settings)

    if isinstance(model, str):
      self.model, _, _ = load(ZhuGo, model)
    else:
      self.model = model
    self.encoder = ZhuGoEncoder()

    self.noise_intensity = noise_intensity
    self.noise = torch.distributions.Dirichlet

  def select_move(self, game_state: GameState) -> Move:
    turn_start_timestamp = time.time()

    board = game_state.board

    self.root = AINode(
      self.model, self.encoder, self.noise_intensity, self.noise,
      game_state=game_state, pool=self.pool, **self.node_settings
    )

    while True:
      self.root.propagate()

      if self.ui is not None:
        self.ui.display_mcts(MCTSData(
          list(self.root.q / 2 + 0.5), # no list would trigger numpy pickling error for unknown reason
          list(self.root.visited_times),
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
