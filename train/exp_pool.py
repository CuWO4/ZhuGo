import torch
from sortedcontainers import SortedList
import pickle
import os
from typing import Callable

__all__ = [
  'Record',
  'ExpPool',
]


class Record:
  __slots__ = ['input', 'policy_target', 'value_target', 'loss', 'decay_coeff']

  def __init__(
    self,
    input: torch.Tensor, policy_target: torch.Tensor, value_target: torch.Tensor,
    loss: float, decay_coeff: float = 1.0
  ):
    '''input(1, C, N, M), policy_target(1, N, M), value_target(1, 1)'''
    self.input: torch.Tensor = input.cpu().detach()
    self.policy_target: torch.Tensor = policy_target.cpu().detach()
    self.value_target: torch.Tensor = value_target.cpu().detach()
    self.loss: float = loss
    self.decay_coeff: float = decay_coeff

  @property
  def priority(self):
    return self.loss * self.decay_coeff
  
  @staticmethod
  def from_tensors(
    inputs: torch.Tensor, # (B, C, N, M)
    policy_targets: torch.Tensor, # (B, N, M)
    value_targets: torch.Tensor, # (B, 1)
    losses: torch.Tensor, # (B, 1)
    decay_coeffs: list[float]
  ) -> list['Record']:
    return [
      Record(input.unsqueeze(0), policy_target.unsqueeze(0), value_target.unsqueeze(0), loss.item(), decay_coeff)
      for input, policy_target, value_target, loss, decay_coeff
      in zip(inputs, policy_targets, value_targets, losses, decay_coeffs)
    ]

class ExpPool:
  def __init__(self, capacity: int, decay_ratio: float = 0.99):
    self.capacity = capacity
    self.decay_ratio: float = decay_ratio
    self.sorted_records: SortedList[Record] = SortedList(key=lambda record: record.priority)

  def insert_record(self, records: list[Record]):
    self.sorted_records.update(records)

    if self.size > self.capacity: # remove redundant records with minimal losses, leave the high losses records
      del self.sorted_records[:self.size - self.capacity]

  def train_on_batch(
    self,
    batch_size: int,
    train_f: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, list[float]], torch.Tensor],
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  ):
    '''
    the function will pass corresponding tensors to train_f, then collect the return losses to update
    internal state. you could define the train_f as a closure function, referencing the model or any
    object you would like to participate in train process.

    train_f(inputs, policy_targets, value_targets, original_losses) -> losses
    where inputs(B, C, N, M), policy_targets(B, N, M), value_targets(B, 1), losses(B)
    '''
    if self.size < batch_size:
      print(f'runtime warning: exp pool too small to get_batch(): {self.size} vs. {batch_size} <{__file__}>')
      batch_size = self.size

    if batch_size == 0:
      print(f'runtime warning: try to get batch with batch_size = 0 <{__file__}>')
      return

    inputs, policy_targets, value_targets, original_losses, decay_coeffs = self._get_batch(batch_size, device=device)


    losses = train_f(inputs, policy_targets, value_targets, original_losses)

    new_records = Record.from_tensors(inputs, policy_targets, value_targets, losses, decay_coeffs)
    self._delete_high_priority_records(batch_size)
    self._priority_decay()
    self.insert_record(new_records)

  @property
  def size(self):
    return len(self.sorted_records)

  @property
  def loss_mean(self):
    if self.size == 0:
      raise ValueError('try to get the loss mean of an empty ExpPool')
    initialized_records = self._initialized_records
    if (len(initialized_records) == 0):
      return 0
    return sum([record.loss for record in initialized_records]) / len(initialized_records)

  @property
  def loss_variance(self):
    if self.size == 0:
      raise ValueError('try to get the loss variance of an empty ExpPool')
    initialized_records = self._initialized_records
    if len(initialized_records) == 0:
      return 0
    mean = self.loss_mean
    return sum([(record.loss - mean) ** 2 for record in initialized_records]) / len(initialized_records)

  @property
  def loss_median(self):
    if self.size == 0:
      raise ValueError('try to get the loss median of an empty ExpPool')
    initialized_records = self._initialized_records
    if len(initialized_records) == 0:
      return 0
    return self.sorted_records[len(initialized_records) // 2].loss

  def save_to_disk(self, file_path: str):
    try:
      os.makedirs(os.path.dirname(file_path), exist_ok=True)
      with open(file_path, 'wb') as f:
        pickle.dump((self.capacity, list(self.sorted_records)), f)
    except (IOError, OSError) as e:
      print(f"ExpPool save failed: {e}")

  @classmethod
  def load_from_disk(cls, file_path: str) -> 'ExpPool':
    try:
      with open(file_path, 'rb') as f:
        capacity, records = pickle.load(f)
    except (IOError, OSError) as e:
      print(f"ExpPool load failed: {e}")

    exp_pool = cls(capacity)
    exp_pool.sorted_records = SortedList(records, key=lambda record: record.loss)
    return exp_pool

  def _get_batch(self, batch_size: int, *, device = 'cuda' if torch.cuda.is_available() else 'cpu') \
    -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float], list[float]]:
    '''
    return inputs(B, N, M), policy_targets(B, N, M), value_targets(B, 1), original_average_loss, decay_coeffs

    remove sampled records. after sampling and training, insert them back by users explicitly
    to update losses
    '''
    top_records = self.sorted_records[-batch_size:]
    inputs = torch.cat([record.input for record in top_records], dim=0).to(device=device)
    policy_targets = torch.cat([record.policy_target for record in top_records], dim=0).to(device=device)
    value_targets = torch.cat([record.value_target for record in top_records], dim=0).to(device=device)
    original_average_loss = [record.loss for record in top_records]
    decay_coeffs = [record.decay_coeff for record in top_records]

    return inputs, policy_targets, value_targets, original_average_loss, decay_coeffs

  def _delete_high_priority_records(self, batch_size: int):
    del self.sorted_records[-batch_size:]

  def _priority_decay(self):
    for record in self.sorted_records:
      record.decay_coeff *= self.decay_ratio
    # no need to readjust since the order relation remains

  @property
  def _initialized_records(self) -> list[Record]:
    return [record for record in self.sorted_records if record.loss < float('inf')]
