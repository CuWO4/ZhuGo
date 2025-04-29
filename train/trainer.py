from ai.manager import ModelManager
from .dataloader import BGTFDataLoader
from .meta import MetaData

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Callable
import time
from datetime import datetime

__all__ = [
  'Trainer'
]


def ctrl_c_catcher(func: Callable, exit_func: Callable):
  try:
    func()
  except KeyboardInterrupt:
    pass
  except Exception as e:
    raise e
  finally:
    while True:
      try:
        exit_func()
        return
      except KeyboardInterrupt:
        pass

def cross_entropy(target: torch.Tensor, output_logits: torch.Tensor) -> torch.Tensor:
  '''p(B, N), q_logits(B, N) -> (B, 1)'''
  assert target.shape == output_logits.shape and len(target.shape) == 2, (
    f'improper shape {target.shape=} vs. {output_logits.shape=}'
  )
  return -torch.sum(
    target * nn.functional.log_softmax(output_logits, dim=-1),
    dim=-1
  ).unsqueeze(1)

def mse(target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
  '''p(B, 1), q(B, 1) -> (B, 1)'''
  assert target.shape == output.shape and len(target.shape) == 2 and target.shape[1] == 1, (
    f'improper shape {target.shape=} vs. {output.shape=}'
  )
  return (target - output) ** 2

class Trainer:
  def __init__(
    self,
    model_manager: ModelManager,
    dataloader: BGTFDataLoader,
    batch_per_test: int,
    test_dataloader: BGTFDataLoader | None = None,
    policy_lost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cross_entropy,
    value_lost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse,
    base_lr: float = 0.1,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    T_max: int = 10000,
    eta_min: float = 1e-3,
    policy_loss_weight: float = 0.7,
    value_loss_weight: float = 0.3,
    checkpoint_interval_sec: int = 3600,
  ):
    self.model_manager: ModelManager = model_manager
    self.dataloader: BGTFDataLoader = dataloader
    self.batch_per_test: int = batch_per_test
    self.test_dataloader: BGTFDataLoader | None = test_dataloader
    self.policy_lost_fn: Callable = policy_lost_fn
    self.value_lost_fn: Callable = value_lost_fn
    self.base_lr: float = base_lr
    self.weight_decay: float = weight_decay
    self.momentum: float = momentum
    self.T_max: int = T_max
    self.eta_min: float = eta_min
    self.policy_loss_weight: float = policy_loss_weight
    self.value_loss_weight: float = value_loss_weight
    self.checkpoint_interval_sec: int = checkpoint_interval_sec

  def train(self, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    if __debug__:
      print(
        'the assertion is on. you may want to execute script '
        'with `-O` to flag to stop dataloader complaining the '
        'noises in dataset are bad moves.'
      )

    with self.model_manager.load_summary_writer() as writer:
      model = self.model_manager.load_model(device = device)
      model.train()
      meta = self.model_manager.load_meta()

      def stop_handling():
        print('stopped. saving...')
        self.model_manager.save_model(model)
        self.model_manager.save_meta(meta)

      ctrl_c_catcher(
        lambda: self.train_body(model, meta, writer),
        stop_handling
      )

  def train_body(self, model: nn.Module, meta: MetaData, writer: SummaryWriter):
    optimizer = optim.SGD(
      model.parameters(),
      lr = self.base_lr,
      weight_decay = self.weight_decay,
      momentum = self.momentum
    )
    schedular = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max = self.T_max, eta_min = self.eta_min
    )

    last_checkpoint_time = time.time()

    for inputs, policy_targets, value_targets in self.dataloader:
      policy_logits, value_logits = model(inputs)

      policy_losses = self.policy_lost_fn(policy_targets, policy_logits)
      value_losses = self.value_lost_fn(value_targets, nn.functional.tanh(value_logits))
      losses = self.policy_loss_weight * policy_losses + self.value_loss_weight * value_losses

      loss = torch.mean(losses)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      schedular.step()

      print(
        f'{"train:":>10}'
        f'{loss.item():12.3f}'
        f'{"press Ctrl-C to stop":>30}'
      )

      writer.add_scalar('train/batch-loss', loss, meta.batches)
      writer.add_scalar('train/lr', schedular.get_last_lr()[0], meta.batches)

      meta.batches += 1

      if meta.batches % self.batch_per_test != 0:
        continue

      if self.test_dataloader is not None:
        loss = self.test_model(model)
        print(
          f'{"test:":>10}'
          f'{loss.item():12.3f}'
        )
        writer.add_scalar('test/loss', loss, meta.batches)

      if time.time() - last_checkpoint_time >= self.checkpoint_interval_sec:
        print('saving checkpoint...')
        self.save_checkpoint(model)
        self.log_histogram(model, meta, writer)
        last_checkpoint_time = time.time()
        print(f'checkpoint saved at {datetime.now().strftime("%H:%M:%S")}')


  def test_model(self, model: nn.Module) -> torch.Tensor:
    '''return loss'''
    assert self.test_dataloader is not None
    inputs, policies, values = next(iter(self.test_dataloader))

    with torch.no_grad():
      model.eval()
      policy_logits, value_logits = model(inputs)
      model.train()

    policy_losses = self.policy_lost_fn(policies, policy_logits)
    value_losses = self.value_lost_fn(values, nn.functional.tanh(value_logits))
    losses = self.policy_loss_weight * policy_losses + self.value_loss_weight * value_losses
    loss = torch.mean(losses)
    return loss

  def save_checkpoint(self, model: nn.Module):
    self.model_manager.save_checkpoint(model)

  def log_histogram(self, model: nn.Module, meta: MetaData, writer: SummaryWriter):
    for name, param in model.named_parameters():
      writer.add_histogram(f'weights/{name}', param, meta.batches)
      if param.grad is not None:
        writer.add_histogram(f'grads/{name}', param.grad, meta.batches)
