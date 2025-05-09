from ai.manager import ModelManager
from .dataloader import BGTFDataLoader
from .optimizer_manager import OptimizerManager
from .meta import MetaData

from ai.encoder.zhugo_encoder import ZhuGoEncoder # dirty code, but let's do it for now

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
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
  finally:
    exit_func()

MAX_LOSS_VALUE = 30 # loss more than this will be clamped, to avoid extreme gradient

def cross_entropy(target: torch.Tensor, output_logits: torch.Tensor) -> torch.Tensor:
  '''p(B, N), q_logits(B, N) -> (B, 1)'''
  assert target.shape == output_logits.shape and len(target.shape) == 2, (
    f'improper shape {target.shape=} vs. {output_logits.shape=}'
  )
  losses = -torch.sum(
    target * nn.functional.log_softmax(output_logits, dim=-1),
    dim=-1
  ).unsqueeze(1)
  return losses.clamp(0, MAX_LOSS_VALUE)

def scalar_cross_entropy(target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
  '''p(B, 1), q(B, 1) -> (B, 1)'''
  assert target.shape == output.shape and len(target.shape) == 2 and target.shape[1] == 1, (
    f'improper shape {target.shape=} vs. {output.shape=}'
  )
  '''assume -1 <= p, q <= 1, make it (-1, 1) one-hot encoding, return cross entropy'''
  assert torch.all((output >= -1) & (output <= 1))
  losses = -(
    (1 - target) / 2 * torch.log((1 - output) / 2 + 1e-5)
    + (1 + target) / 2 * torch.log((1 + output) / 2 + 1e-5)
  )
  return losses.clamp(0, MAX_LOSS_VALUE)

class Trainer:
  def __init__(
    self,
    *,
    model_manager: ModelManager,
    optimizer_manager: OptimizerManager,
    dataloader: BGTFDataLoader,
    batch_accumulation: int,
    batch_per_test: int,
    test_dataloader: BGTFDataLoader | None = None,
    policy_lost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cross_entropy,
    value_lost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = scalar_cross_entropy,
    gradient_clip: float,
    T_max: int,
    eta_min: float,
    policy_loss_weight: float,
    value_loss_weight: float,
    soft_target_nominal_weight: float,
    softening_intensity: float,
    checkpoint_interval_sec: int,
  ):
    self.model_manager: ModelManager = model_manager
    self.optimizer_manager: OptimizerManager = optimizer_manager
    self.dataloader: BGTFDataLoader = dataloader
    self.batch_accumulation: int = batch_accumulation
    self.batch_per_test: int = batch_per_test
    self.test_dataloader: BGTFDataLoader | None = test_dataloader
    self.policy_lost_fn: Callable = policy_lost_fn
    self.value_lost_fn: Callable = value_lost_fn
    self.gradient_clip: float = gradient_clip
    self.T_max: int = T_max
    self.eta_min: float = eta_min
    self.policy_loss_weight: float = policy_loss_weight
    self.value_loss_weight: float = value_loss_weight
    self.soft_target_nominal_weight: float = soft_target_nominal_weight
    self.softening_intensity: float = softening_intensity
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
      optimizer = self.optimizer_manager.load_optimizer(model.parameters())
      meta = self.model_manager.load_meta()

      def stop_handling():
        print('stopped. saving...')
        self.model_manager.save_model(model)
        self.model_manager.save_meta(meta)
        self.optimizer_manager.save_optimizer(optimizer)

      ctrl_c_catcher(
        lambda: self.train_body(model, optimizer, meta, writer),
        stop_handling
      )

  def train_body(self, model: nn.Module, optimizer: optim.Optimizer, meta: MetaData, writer: SummaryWriter):
    schedular = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max = self.T_max, eta_min = self.eta_min
    )
    scaler = amp.GradScaler()

    last_checkpoint_time = time.time()

    # only for displaying statics
    accumulated_total_loss: float = 0
    accumulated_policy_loss: float = 0
    accumulated_value_loss: float = 0

    begin_batches = meta.batches

    for inputs, policy_targets, value_targets in self.dataloader:
      valid_mask = self.get_valid_mask(inputs)
      policy_targets *= valid_mask # invalid moves does not engage in backward

      with amp.autocast():
        policy_logits, value_logits = model(inputs)

        policy_losses = self.policy_lost_fn(policy_targets, policy_logits)
        value_losses = self.value_lost_fn(value_targets, nn.functional.tanh(value_logits))

        softened_policy_target = policy_targets ** self.softening_intensity
        softened_policy_target /= softened_policy_target.sum(dim = -1, keepdim = True) + 1e-8

        policy_losses += self.soft_target_nominal_weight * self.policy_lost_fn(softened_policy_target, policy_logits)
        policy_losses /= (1 + self.soft_target_nominal_weight)

        losses = self.policy_loss_weight * policy_losses + self.value_loss_weight * value_losses

        loss = torch.mean(losses)
        backward_loss = loss / self.batch_accumulation

      scaler.scale(backward_loss).backward()

      self.log_losses(
        'train', meta,
        loss.item(),
        policy_losses.mean().item(),
        value_losses.mean().item(),
        writer
      )

      writer.add_scalar('train/lr', schedular.get_last_lr()[0], meta.batches)

      accumulated_total_loss += loss.item() / self.batch_accumulation
      accumulated_policy_loss += torch.mean(policy_losses).item() / self.batch_accumulation
      accumulated_value_loss += torch.mean(value_losses).item() / self.batch_accumulation

      if (
        meta.batches - begin_batches > 0
        and (meta.batches - begin_batches) % self.batch_accumulation == self.batch_accumulation - 1
      ):
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        schedular.step()

        if self.batch_accumulation > 0:
          self.log_losses(
            'acc', meta,
            accumulated_total_loss,
            accumulated_policy_loss,
            accumulated_value_loss,
            writer
          )

        accumulated_total_loss = 0
        accumulated_policy_loss = 0
        accumulated_value_loss = 0

      if (
        self.test_dataloader is not None
        and meta.batches - begin_batches > 0
        and (meta.batches - begin_batches) % self.batch_per_test == self.batch_per_test - 1
      ):
        validate_policy_loss, validate_value_loss = self.test_model(model)
        validate_loss = (
          self.policy_loss_weight * validate_policy_loss
          + self.value_loss_weight * validate_value_loss
        )
        self.log_losses(
          'test', meta,
          validate_loss.item(),
          validate_policy_loss.item(),
          validate_value_loss.item(),
          writer
        )

      meta.batches += 1

      if time.time() - last_checkpoint_time >= self.checkpoint_interval_sec:
        print('saving checkpoint...')
        self.save_checkpoint(model)
        self.log_histogram(model, meta, writer)
        last_checkpoint_time = time.time()
        print(f'checkpoint saved at {datetime.now().strftime("%H:%M:%S")}')

  def test_model(self, model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    '''return policy_loss(1), value_loss(1)'''
    assert self.test_dataloader is not None
    inputs, policies, values = next(iter(self.test_dataloader))

    with torch.no_grad():
      model.eval()
      policy_logits, value_logits = model(inputs)
      model.train()

    policy_losses = self.policy_lost_fn(policies, policy_logits)
    value_losses = self.value_lost_fn(values, nn.functional.tanh(value_logits))
    return policy_losses.mean(), value_losses.mean()

  def save_checkpoint(self, model: nn.Module):
    self.model_manager.save_checkpoint(model)

  @staticmethod
  def log_losses(
    tag: str,
    meta: MetaData,
    total_loss: float,
    policy_loss: float,
    value_loss: float,
    writer: SummaryWriter,
  ):
    print(
      f'{meta.batches:>8}'
      f'{f"<{tag}>":>15}'
      f'{total_loss:12.3f}'
    )
    writer.add_scalars('train/total_loss', { tag: total_loss, }, meta.batches)
    writer.add_scalars('train/policy_loss', { tag: policy_loss, }, meta.batches)
    writer.add_scalars('train/value_loss', { tag: value_loss, }, meta.batches)

  @staticmethod
  def log_histogram(model: nn.Module, meta: MetaData, writer: SummaryWriter):
    for name, param in model.named_parameters():
      writer.add_histogram(f'weights/{name}', param, meta.batches)
      if param.grad is not None:
        writer.add_histogram(f'grads/{name}', param.grad, meta.batches)

  @staticmethod
  def get_valid_mask(inputs: torch.Tensor) -> torch.Tensor:
    '''
    inputs(B, C, 19, 19) -> (B, 362)
    assume inputs is encoded with ZhuGoEncoder
    '''
    batch_size = inputs.shape[0]
    valid_mask = inputs[:, ZhuGoEncoder.VALID_MOVE_OFF, :, :]
    return torch.cat(
      (
        valid_mask.reshape(batch_size, 361),
        torch.ones(batch_size, 1, device = valid_mask.device)
      ), dim = 1
    )
