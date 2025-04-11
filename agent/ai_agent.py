from agent.mcts.base import Node
from .mcts_agent import MCTSAgent

from ai.zhugo import ZhuGo
from ai.encoder.zhugo_encoder import ZhuGoEncoder
from ai.manager import load, load_deployed_model
from .mcts.ai_node import AINode
from go.goboard import GameState
from utils.mcts_data import MCTSData
from .mcts.utils import best_move_idx

import torch

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

    self.encoder = ZhuGoEncoder()

    if isinstance(model, str):
      try: # try to load deployed model
        warmup_dumb_input = self.encoder.encode(GameState.new_game()).unsqueeze(0)
        self.model = load_deployed_model(model, warmup_dumb_input = warmup_dumb_input)
      except Exception: # failed, load native pytorch model instead
        self.model = load(ZhuGo, model)
      self.model.eval()
    else:
      assert isinstance(self.model, ZhuGo)
      self.model = model

  def update_ui_mcts(self):
    assert self.ui is not None
    self.ui.display_mcts(MCTSData(
      list(self.root.q / 2 + 0.5),
      # for unknown reasons, not converting to a list
      # will trigger a serialization error
      list(self.root.visited_times),
      best_move_idx(self.root.visited_times, self.root.q),
      self.root.win_rate,
      self.root.game_state.board.size
    ))

  def construct_root(self, game_state: GameState) -> Node:
    return AINode(
      self.model, self.encoder,
      game_state=game_state, pool=self.pool, **self.node_settings
    )
