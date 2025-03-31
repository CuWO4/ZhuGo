import torch
import pickle
from typing import Optional
from sortedcontainers import SortedList
import os

__all__ = [
  'Record',
  'ExpPool',
]


class Record:
  __slots__ = ['input_tensor', 'policy_target', 'value_target', 'loss', 'priority']

  def __init__(self, input_tensor: torch.Tensor, policy_target: torch.Tensor, value_target: torch.Tensor, loss: float):
    '''input_tensor(1, C, N, M), policy_target(1, N, M), value_target(1, 1)'''
    self.input_tensor = input_tensor.cpu().detach()
    self.policy_target = policy_target.cpu().detach()
    self.value_target = value_target.cpu().detach()
    self.loss = loss
    self.priority = self.loss

class ExpPool:
  def __init__(self, capacity: int, decay_ratio: float = 0.99):
    self.capacity = capacity
    self.decay_ratio: float = decay_ratio
    self.sorted_records: SortedList[Record] = SortedList(key=lambda record: record.priority)

  def insert_record(self, records: list[Record]):
    self.sorted_records.update(records)

    if self.size > self.capacity: # remove redundant records with minimal losses, leave the high losses records
      del self.sorted_records[:self.size - self.capacity]

  def insert_tensor(self, inputs: torch.Tensor, policy_targets: torch.Tensor, value_targets: torch.Tensor, losses: torch.Tensor):
    '''inputs(B, C, N, M), policy_targets(B, N, M), value_targets(B, 1), losses(B, 1)'''
    inputs = inputs.cpu()
    policy_targets = policy_targets.cpu()
    value_targets = value_targets.cpu()
    losses = losses.cpu()

    batch_size = inputs.shape[0]
    assert policy_targets.shape[0] == batch_size, \
      f"policy target batch size mismatch {policy_targets.shape[0]} vs. {batch_size}"
    assert value_targets.shape[0] == batch_size, \
      f"value target batch size mismatch {value_targets.shape[0]} vs. {batch_size}"
    assert len(losses) == batch_size, \
      f"losses batch size mismatch {len(losses)} vs. {batch_size}"

    records = [
      Record(input.unsqueeze(0), policy_target.unsqueeze(0), value_target.unsqueeze(0), loss.item())
      for input, policy_target, value_target, loss in zip(inputs, policy_targets, value_targets, losses)
    ]

    self.insert_record(records)

  def get_batch(self, batch_size: int, *, remove: bool = True, device = 'cuda' if torch.cuda.is_available() else 'cpu') \
    -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float]:
    '''
    return inputs(B, N, M), policy_targets(B, N, M), value_targets(B, 1), original_average_loss

    by default, remove sampled records. after sampling and training, insert them back by users explicitly
    to update losses
    '''

    if self.size < batch_size:
      print(f'runtime warning: exp pool too small to get_batch(): {self.size} vs. {batch_size} <{__file__}>')
      batch_size = self.size

    if batch_size == 0:
      print(f'runtime warning: try to get batch with batch_size = 0 <{__file__}>')
      return (torch.tensor([]), torch.tensor([]), torch.tensor([]))

    top_records = self.sorted_records[-batch_size:]
    input_tensors = torch.cat([record.input_tensor for record in top_records], dim=0).to(device=device)
    policy_targets = torch.cat([record.policy_target for record in top_records], dim=0).to(device=device)
    value_targets = torch.cat([record.value_target for record in top_records], dim=0).to(device=device)
    original_average_loss = sum([record.loss for record in top_records]) / len(top_records)

    if remove:
      del self.sorted_records[-batch_size:]

    self.priority_decay()

    return input_tensors, policy_targets, value_targets, original_average_loss

  def priority_decay(self):
    for record in self.sorted_records:
      record.priority *= self.decay_ratio
    # no need readjust since order relation remain

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

  @property
  def size(self):
    return len(self.sorted_records)

  @property
  def initialized_records(self) -> list[Record]:
    return [record for record in self.sorted_records if record.loss < float('inf')]

  @property
  def loss_mean(self):
    if len(self.sorted_records) == 0:
      raise ValueError('try to get the loss mean of an empty ExpPool')
    initialized_records = self.initialized_records
    if (len(initialized_records) == 0):
      return 0
    return sum([record.loss for record in initialized_records]) / len(initialized_records)

  @property
  def loss_variance(self):
    if len(len(self.sorted_records)) == 0:
      raise ValueError('try to get the loss variance of an empty ExpPool')
    initialized_records = self.initialized_records
    if len(initialized_records) == 0:
      return 0
    mean = self.loss_mean
    return sum([(record.loss - mean) ** 2 for record in initialized_records]) / len(initialized_records)

  @property
  def loss_median(self):
    if len(self.sorted_records) == 0:
      raise ValueError('try to get the loss median of an empty ExpPool')
    initialized_records = self.initialized_records
    if len(initialized_records) == 0:
      return 0
    return self.sorted_records[len(initialized_records) // 2].loss