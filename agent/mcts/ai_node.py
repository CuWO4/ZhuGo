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

  def __sub__(self, other: 'PredictResult') -> 'PredictResult':
    return PredictResult(
      self.player,
      self.q - other.get_q(self.player)
    )

class AINode(Node):
  def __init__(
    self, model: ZhuGo, encoder: ZhuGoEncoder,
    *, game_state: GameState, pool: mp.Pool, c: float = 1.0,
    evaluated_states: dict | None = None
  ):
    super().__init__(game_state=game_state, pool=pool, c=c)

    self.model = model
    self.encoder = encoder

    with torch.no_grad():
      input_tensor = encoder.encode(game_state).unsqueeze(0)
      policy_logits, value_logits, _, _ = model(input_tensor)
      policy_logits = policy_logits.detach().squeeze(0)
      value_logits = value_logits.detach()

    policy_logits += (torch.tensor(self.legal_mask, device=policy_logits.device) - 1) * 1e5

    self.policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()
    self.value = torch.tanh(value_logits.detach().cpu())[0, 0].item()

    self.is_leaf = True

    self._visited_times: np.ndarray = np.zeros(self.policy_size)

    self.leaf_q_sum: np.ndarray = np.zeros(self.policy_size)
    self.leaf_count: np.ndarray = np.zeros(self.policy_size)

    self.init_q_mask: np.ndarray = np.array([self.value] * self.policy_size)

    self.evaluated_states: dict = evaluated_states if evaluated_states is not None else { game_state: self }

    self.parents: list[tuple['AINode', int]] = []

  def propagate(self) -> None:
    if self.game_state.is_over():
      self.update_and_notify_parents(
        None, 1,
        PredictResult(
          self.game_state.next_player,
          (1 if self.game_state.winner() == self.game_state.next_player else -1)
        ), 0
      )
      return

    if self.legal_play_count > 0:
      move_idx = exploring_move_indexes(self.ucb, 1)[0]
    else:
      move_idx = move_to_idx(Move.pass_turn(), self.game_state.board.size)

    if self.branches[move_idx] is not None:
      self.branches[move_idx].propagate()
      return

    move = idx_to_move(move_idx, self.game_state.board.size)
    new_branch = self.branch(move)
    if new_branch.is_leaf:
      new_q_sum_acc = new_branch.wrap_to_result()
      nr_new_visited_times = 1
      nr_new_leaves = 1
    else:
      new_q_sum_acc = PredictResult(
        new_branch.game_state.next_player,
        np.sum(new_branch.leaf_q_sum)
      )
      nr_new_visited_times = np.sum(new_branch.visited_times)
      nr_new_leaves = np.sum(new_branch.leaf_count)

    if self.is_leaf:
      new_q_sum_acc -= self.wrap_to_result()
      nr_new_leaves -= 1

    self.update_and_notify_parents(move_idx, nr_new_visited_times, new_q_sum_acc, nr_new_leaves)

    self.is_leaf = False

  def add_parent(self, parent: 'AINode', idx: int):
    self.parents.append((parent, idx))

  def branch(self, move: Move) -> 'AINode':
    move_idx = move_to_idx(move, self.game_state.board.size)
    if self.branches[move_idx] is None:
      new_state = self.game_state.apply_move(move)
      if self.evaluated_states.get(new_state) is None:
        self.evaluated_states[new_state] = AINode(
          self.model, self.encoder,
          game_state=new_state,
          pool=self.pool, c=self.c,
          evaluated_states=self.evaluated_states
        )
      self.branches[move_idx] = self.evaluated_states[new_state]
      self.branches[move_idx].add_parent(self, move_idx)
    return self.branches[move_idx]

  def switch_branch(self, move: Move) -> 'AINode':
    return self.branch(move)

  def update_and_notify_parents(
    self, child_idx: int | None, nr_new_visited_times: int,
    leaf_q_sum_acc: PredictResult, nr_new_leaves: int
  ):
    if child_idx is not None:
      self._visited_times[child_idx] += nr_new_visited_times
      self.leaf_q_sum[child_idx] += leaf_q_sum_acc.get_q(self.game_state.next_player)
      self.leaf_count[child_idx] += nr_new_leaves
    for parent, idx in self.parents:
      parent.update_and_notify_parents(idx, nr_new_visited_times, leaf_q_sum_acc, nr_new_leaves)

  def wrap_to_result(self) -> PredictResult:
    return PredictResult(self.game_state.next_player, self.value)

  @property
  def visited_times(self) -> np.ndarray:
    return self._visited_times

  @property
  def win_rate(self) -> float:
    if np.sum(self.visited_times) == 0:
      return self.value
    else:
      return (
        self.value if self.is_leaf
        else (np.sum(self.q * self.visited_times) / np.sum(self.visited_times) / 2) + 0.5
      )

  @property
  def q(self) -> np.ndarray:
    return np.where(self.leaf_count > 0, self.leaf_q_sum / (self.leaf_count + 1e-5), self.init_q_mask)

  @property
  def ucb(self) -> np.ndarray:
    return (
      self.q
      + self.c * np.sqrt(1 + np.sum(self.visited_times)) / (1 + self.visited_times) * self.policy
      + (self.legal_mask - 1) * 1e5
    )
