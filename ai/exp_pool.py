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
  __slots__ = ['input_tensor', 'policy_target', 'value_target', 'loss']
  
  def __init__(self, input_tensor: torch.Tensor, policy_target: torch.Tensor, value_target: torch.Tensor, loss: float):
    self.input_tensor = input_tensor.cpu().detach()
    self.policy_target = policy_target.cpu().detach() if policy_target is not None else None
    self.value_target = value_target.cpu().detach() if value_target is not None else None
    self.loss = loss

class ExpPool:
  def __init__(self, capacity: int):
    self.capacity = capacity
    self.sorted_records = SortedList(key=lambda record: record.loss)

  def insert_record(self, records: list[Record]):
    self.sorted_records.update(records)

    if self.size > self.capacity: # remove redundant records with minimal losses, leave the high losses records
      del self.sorted_records[:self.size - self.capacity]
      
  def insert_tensor(self, inputs: torch.Tensor, policy_targets: torch.Tensor, value_targets: torch.Tensor, losses: torch.Tensor):
    inputs = inputs.cpu()
    if policy_targets is None: policy_targets = torch.tensor([0] * len(inputs), device='cpu')
    else: policy_targets = policy_targets.cpu()
    if value_targets is None: value_targets = torch.tensor([0] * len(inputs), device='cpu')
    else: value_targets = value_targets.cpu()
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

  def get_batch(self, batch_size: int, *, remove: bool = True, need_policy: bool = True, need_value: bool = True,
    device = 'cuda' if torch.cuda.is_available() else 'cpu') \
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
      return (
        torch.tensor([]), 
        torch.tensor([]) if need_policy else None, 
        torch.tensor([]) if need_policy else None
      )

    top_records = self.sorted_records[-batch_size:]
    input_tensors = torch.cat([record.input_tensor for record in top_records], dim=0).to(device=device)
    policy_targets = torch.cat([record.policy_target for record in top_records], dim=0).to(device=device) if need_policy else None
    value_targets = torch.cat([record.value_target for record in top_records], dim=0).to(device=device) if need_value else None
    original_average_loss = sum([record.loss for record in top_records]) / len(top_records)

    if remove:
      del self.sorted_records[-batch_size:]

    return input_tensors, policy_targets, value_targets, original_average_loss

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
    initialized_records = self.initialized_records
    if len(initialized_records) == 0:
      raise ValueError('try to get the loss mean of an empty ExpPool')
    return sum([record.loss for record in initialized_records]) / len(initialized_records)
  
  @property
  def loss_variance(self):
    initialized_records = self.initialized_records
    if len(initialized_records) == 0:
      raise ValueError('try to get the loss variance of an empty ExpPool')
    mean = self.loss_mean
    return sum([(record.loss - mean) ** 2 for record in initialized_records]) / len(initialized_records)
  
  @property
  def loss_median(self):
    initialized_records = self.initialized_records
    if len(initialized_records) == 0:
      raise ValueError('try to get the loss median of an empty ExpPool')
    return self.sorted_records[len(initialized_records) // 2].loss