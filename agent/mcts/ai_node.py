import multiprocessing as mp
from .base import Node

from ai.zhugo import ZhuGo
from ai.encoder.zhugo_encoder import ZhuGoEncoder

from .utils import exploring_move_indexes

from go.goboard import GameState, Move
from go.gotypes import Player
from utils.move_idx_transformer import idx_to_move, move_to_idx

import torch
import numpy as np
import multiprocessing as mp

__all__ = [
  'AINode',
]


class PredictResult:
  __slots__ = ['player', 'q']
  def __init__(self, player: Player, q: float):
    self.player = player
    self.q = q

  def get_q(self, player: Player) -> float:
    return self.q if player == self.player else -self.q


class AINode(Node):
  def __init__(
    self, model: ZhuGo, encoder: ZhuGoEncoder,
    *, game_state: GameState, pool: mp.Pool, c: float = 1.0
  ):
    super().__init__(game_state=game_state, pool=pool, c=c)

    self.model = model
    self.encoder = encoder

    with torch.no_grad():
      input_tensor = encoder.encode(game_state).unsqueeze(0)
      policy_logits, value_logits = model(input_tensor)
      policy_logits = policy_logits.detach().squeeze(0)
      value_logits = value_logits.detach()

    policy_logits += (torch.tensor(self.legal_mask, device=policy_logits.device) - 1) * 1e5

    self.policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()
    self.value = torch.tanh(value_logits.detach().cpu())[0, 0].item()

    self.is_leaf = True

    self._visited_times: np.ndarray = np.zeros(self.policy_size)

    self.leaf_q_sum: np.ndarray = np.zeros(self.policy_size)
    self.leaf_count = np.ndarray = np.zeros(self.policy_size)

  def propagate(self) -> tuple[PredictResult, PredictResult | None]:
    '''return (new_result, removed_result)'''

    if self.game_state.is_over():
      return (
        PredictResult(
          self.game_state.next_player,
          (1 if self.game_state.winner() == self.game_state.next_player else -1)
        ),
        None
      )

    if self.legal_play_count > 0:
      move_idx = exploring_move_indexes(self.ucb, 1)[0]
    else:
      move_idx = move_to_idx(Move.pass_turn(), self.game_state.board.size)

    if self.branches[move_idx] is None:
      move = idx_to_move(move_idx, self.game_state.board.size)
      new_result = self.branch(move).wrap_to_result()
      self.leaf_count[move_idx] += 1

      removed_result = self.wrap_to_result() if self.is_leaf else None

    else: # self.branches[move_idx] is not None
      new_result, removed_result = self.branches[move_idx].propagate()

    self._visited_times[move_idx] += 1
    self.leaf_q_sum[move_idx] += new_result.get_q(self.game_state.next_player)

    if removed_result is not None:
      if not self.is_leaf:
        self.leaf_q_sum[move_idx] -= removed_result.get_q(self.game_state.next_player)
    else:
      self.leaf_count[move_idx] += 1

    self.is_leaf = False

    return new_result, removed_result

  def branch(self, move: Move) -> 'AINode':
    move_idx = move_to_idx(move, self.game_state.board.size)
    if self.branches[move_idx] is None:
      self.branches[move_idx] = AINode(
        self.model, self.encoder,
        game_state=self.game_state.apply_move(move),
        pool=self.pool, c=self.c
      )
    return self.branches[move_idx]

  def switch_branch(self, move: Move) -> 'AINode':
    return self.branch(move)

  def wrap_to_result(self) -> PredictResult:
    return PredictResult(self.game_state.next_player, self.value)

  @property
  def visited_times(self) -> np.ndarray:
    return self._visited_times

  @property
  def win_rate(self) -> float:
    if np.sum(self.visited_times) == 0:
      return 0.5
    else:
      return (
        self.value if self.is_leaf
        else (np.sum(self.q * self.visited_times) / np.sum(self.visited_times) / 2) + 0.5
      )

  @property
  def q(self) -> np.ndarray:
    return self.leaf_q_sum / (self.leaf_count + 1e-5)

  @property
  def ucb(self) -> np.ndarray:
    return (
      self.q
      + self.c * np.sqrt(1 + np.sum(self.visited_times)) / (1 + self.visited_times) * self.policy
      + (self.legal_mask - 1) * 1e5
    )
