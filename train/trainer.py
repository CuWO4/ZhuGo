from ai.manager import ModelManager
from .dataloader import BGTFDataLoader
from .optimizer_manager import OptimizerManager
from .meta import MetaData
from ai.zhugo import ZhuGoValueHead

from ai.encoder.zhugo_encoder import ZhuGoEncoder # dirty code, but let's do it for now

import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Iterable, Optional
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

def point_wise_scalar_cross_entropy(target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
  '''p(B, N), q(B, N) -> (B, 1)'''
  assert target.shape == output.shape and len(target.shape) == 2, (
    f'improper shape {target.shape=} vs. {output.shape=}'
  )
  # assume -1 <= p, q <= 1, make it (-1, 1) one-hot encoding, return cross entropy
  assert torch.all((target >= -1) & (target <= 1) & (output >= -1) & (output <= 1))
  losses = -torch.mean(
    (1 - target) / 2 * torch.log((1 - output) / 2 + 1e-5)
    + (1 + target) / 2 * torch.log((1 + output) / 2 + 1e-5),
    dim=-1, keepdim=True
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
    policy_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cross_entropy,
    win_rate_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = point_wise_scalar_cross_entropy,
    ownership_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = point_wise_scalar_cross_entropy,
    score_dist_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cross_entropy,
    score_mean_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.HuberLoss(delta=10, reduction='none'),
    gradient_clip: float,
    policy_loss_weight: Optional[float] = None,
    win_rate_loss_weight: float,
    softened_policy_loss_weight: float,
    softening_intensity: float,
    ownership_loss_weight: float,
    score_dist_loss_weight: float,
    score_mean_loss_weight: float,
    checkpoint_interval_sec: int,
  ):
    self.model_manager: ModelManager = model_manager
    self.optimizer_manager: OptimizerManager = optimizer_manager
    self.dataloader: BGTFDataLoader = dataloader
    self.batch_accumulation: int = batch_accumulation
    self.batch_per_test: int = batch_per_test
    self.test_dataloader: Iterable[BGTFDataLoader] | None = iter(test_dataloader) \
      if test_dataloader is not None else None
    self.policy_loss_fn: Callable = policy_loss_fn
    self.win_rate_loss_fn: Callable = win_rate_loss_fn
    self.ownership_loss_fn: Callable = ownership_loss_fn
    self.score_dist_loss_fn: Callable = score_dist_loss_fn
    self.score_mean_loss_fn: Callable = score_mean_loss_fn
    if policy_loss_weight is not None:
      print('<Trainer> policy_loss_weight is deprecated, which will not take in counter anymore. '
            'set win_rate_loss_weight only instead, while equivalent policy_loss_weight = 1.0.')
    self.gradient_clip: float = gradient_clip
    self.win_rate_loss_weight: float = win_rate_loss_weight
    self.softened_policy_loss_weight: float = softened_policy_loss_weight
    self.softening_intensity: float = softening_intensity
    self.ownership_loss_weight: float = ownership_loss_weight
    self.score_dist_loss_weight: float = score_dist_loss_weight
    self.score_mean_loss_weight: float = score_mean_loss_weight
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
    scaler = amp.GradScaler()

    last_checkpoint_time = time.time()

    begin_batches = meta.batches

    for data in self.dataloader:
      with amp.autocast('cuda'):
        data = self.execute_model(model, data)
        policy_loss = data['policy_loss']
        win_rate_loss = data['win_rate_loss']
        softened_policy_loss = data['softened_policy_loss']
        ownership_loss = data['ownership_loss']
        score_dist_loss = data['score_dist_loss']
        score_mean_loss = data['score_mean_loss']
        loss = data['loss']
        policy_accuracy = data['policy_accuracy']
        win_rate_accuracy = data['win_rate_accuracy']
        ownership_accuracy = data['ownership_accuracy']
        backward_loss = loss / self.batch_accumulation

      scaler.scale(backward_loss).backward()

      self.log_losses(
        tag = 'train',
        meta = meta,
        total_loss = loss,
        policy_loss = policy_loss,
        win_rate_loss = win_rate_loss,
        softened_policy_loss = softened_policy_loss,
        ownership_loss = ownership_loss,
        score_dist_loss = score_dist_loss,
        score_mean_loss = score_mean_loss,
        policy_accuracy = policy_accuracy,
        win_rate_accuracy = win_rate_accuracy,
        ownership_accuracy = ownership_accuracy,
        writer = writer
      )

      if (
        meta.batches - begin_batches > 0
        and (meta.batches - begin_batches) % self.batch_accumulation == self.batch_accumulation - 1
      ):
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

      if (
        self.test_dataloader is not None
        and meta.batches - begin_batches > 0
        and (meta.batches - begin_batches) % self.batch_per_test == self.batch_per_test - 1
      ):
        with torch.no_grad():
          model.eval()
          data = self.execute_model(model, next(self.test_dataloader))
          policy_loss = data['policy_loss']
          win_rate_loss = data['win_rate_loss']
          softened_policy_loss = data['softened_policy_loss']
          ownership_loss = data['ownership_loss']
          score_dist_loss = data['score_dist_loss']
          score_mean_loss = data['score_mean_loss']
          policy_accuracy = data['policy_accuracy']
          win_rate_accuracy = data['win_rate_accuracy']
          ownership_accuracy = data['ownership_accuracy']
          loss = data['loss']
          model.train()
        self.log_losses(
          tag = 'test',
          meta = meta,
          total_loss = loss,
          policy_loss = policy_loss,
          win_rate_loss = win_rate_loss,
          softened_policy_loss = softened_policy_loss,
          ownership_loss = ownership_loss,
          score_dist_loss = score_dist_loss,
          score_mean_loss = score_mean_loss,
          policy_accuracy = policy_accuracy,
          win_rate_accuracy = win_rate_accuracy,
          ownership_accuracy = ownership_accuracy,
          writer = writer
        )

      meta.batches += 1

      if time.time() - last_checkpoint_time >= self.checkpoint_interval_sec:
        print('saving checkpoint...')
        self.save_checkpoint(model)
        self.log_histogram(model, meta, writer)
        last_checkpoint_time = time.time()
        print(f'checkpoint saved at {datetime.now().strftime("%H:%M:%S")}')

  def execute_model(self, model: nn.Module, data: tuple[torch.Tensor]) -> dict[str, torch.Tensor]:
    input_tensor, policy_targets, win_rate_targets, ownership_targets, score_targets = data
    valid_mask = self.get_valid_mask(input_tensor)
    policy_targets = policy_targets * valid_mask # invalid moves does not engage in backward

    max_score = input_tensor.shape[-2] * input_tensor.shape[-1]
    normalized_scores = score_targets / max_score
    score_bucket_target = self.get_score_bucket_target(normalized_scores)

    softened_policy_targets = policy_targets ** self.softening_intensity
    softened_policy_targets /= softened_policy_targets.sum(dim = -1, keepdim = True) + 1e-8

    policy_logits, win_rate_logits, ownership_logits, score_logits = model(input_tensor)

    normalized_score_prediction = torch.softmax(score_logits, dim=-1)

    policy_losses = self.policy_loss_fn(policy_targets, policy_logits)
    win_rate_losses = self.win_rate_loss_fn(win_rate_targets, nn.functional.tanh(win_rate_logits))
    softened_policy_losses = self.policy_loss_fn(softened_policy_targets, policy_logits)
    ownership_losses = self.ownership_loss_fn(ownership_targets, nn.functional.tanh(ownership_logits))
    score_dist_losses = self.score_dist_loss_fn(score_bucket_target, score_logits)
    score_mean_losses = self.score_mean_loss_fn(score_targets, self.get_score_mean(normalized_score_prediction) * max_score)

    policy_loss = policy_losses.mean()
    win_rate_loss = win_rate_losses.mean()
    softened_policy_loss = softened_policy_losses.mean()
    ownership_loss = ownership_losses.mean()
    score_dist_loss = score_dist_losses.mean()
    score_mean_loss = score_mean_losses.mean()

    policy_accuracy = (torch.argmax(policy_logits, dim=-1) == torch.argmax(policy_targets, dim=-1)).float().mean()
    win_rate_accuracy = (torch.sign(win_rate_logits) == torch.sign(win_rate_targets)).float().mean()
    ownership_accuracy = (torch.sign(ownership_logits) == torch.sign(ownership_targets)).float().mean()

    loss = (
      policy_loss
      + self.win_rate_loss_weight * win_rate_loss
      + self.softened_policy_loss_weight * softened_policy_loss
      + self.ownership_loss_weight * ownership_loss
      + self.score_dist_loss_weight * score_dist_loss
      + self.score_mean_loss_weight * score_mean_loss
    )

    return {
      'policy_loss': policy_loss,
      'win_rate_loss': win_rate_loss,
      'softened_policy_loss': softened_policy_loss,
      'ownership_loss': ownership_loss,
      'score_dist_loss': score_dist_loss,
      'score_mean_loss': score_mean_loss,
      'loss': loss,
      'policy_accuracy': policy_accuracy,
      'win_rate_accuracy': win_rate_accuracy,
      'ownership_accuracy': ownership_accuracy,
    }

  def save_checkpoint(self, model: nn.Module):
    self.model_manager.save_checkpoint(model)

  @staticmethod
  def log_losses(
    *,
    tag: str,
    meta: MetaData,
    total_loss: torch.Tensor,
    policy_loss: torch.Tensor,
    win_rate_loss: torch.Tensor,
    softened_policy_loss: torch.Tensor,
    ownership_loss: torch.Tensor,
    score_dist_loss: torch.Tensor,
    score_mean_loss: torch.Tensor,
    policy_accuracy: torch.Tensor,
    win_rate_accuracy: torch.Tensor,
    ownership_accuracy: torch.Tensor,
    writer: SummaryWriter,
  ):
    total_loss = total_loss.item()
    policy_loss = policy_loss.item()
    win_rate_loss = win_rate_loss.item()
    softened_policy_loss = softened_policy_loss.item()
    ownership_loss = ownership_loss.item()
    score_dist_loss = score_dist_loss.item()
    score_mean_loss = score_mean_loss.item()
    policy_accuracy = policy_accuracy.item()
    win_rate_accuracy = win_rate_accuracy.item()
    ownership_accuracy = ownership_accuracy.item()

    print(
      f'{meta.batches:>8}'
      f'{f"<{tag}>":>15}'
      f'{total_loss:12.3f}'
    )
    writer.add_scalars('train/total_loss', { tag: total_loss, }, meta.batches)
    writer.add_scalars('train/policy_loss', { tag: policy_loss, }, meta.batches)
    writer.add_scalars('train/win_rate_loss', { tag: win_rate_loss, }, meta.batches)
    writer.add_scalars('train/softened_policy_loss', { tag: softened_policy_loss }, meta.batches)
    writer.add_scalars('train/ownership_loss', { tag: ownership_loss }, meta.batches)
    writer.add_scalars('train/score_dist_loss', { tag: score_dist_loss }, meta.batches)
    writer.add_scalars('train/score_mean_loss', { tag: score_mean_loss }, meta.batches)
    writer.add_scalars('train/policy_accuracy', { tag: policy_accuracy }, meta.batches)
    writer.add_scalars('train/win_rate_accuracy', { tag: win_rate_accuracy }, meta.batches)
    writer.add_scalars('train/ownership_accuracy', { tag: ownership_accuracy }, meta.batches)

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

  @staticmethod
  def get_score_bucket_target(normalized_scores: torch.Tensor) -> torch.Tensor:
    assert torch.all((normalized_scores >= -1) & (normalized_scores <= 1))
    lower_bounds = ZhuGoValueHead.SCORE_BUCKETS[:, 0]
    upper_bounds = ZhuGoValueHead.SCORE_BUCKETS[:, 1]
    mask_prev = (normalized_scores >= lower_bounds[:-1]) & (normalized_scores < upper_bounds[:-1])
    mask_last = (normalized_scores >= lower_bounds[-1]) & (normalized_scores <= upper_bounds[-1])
    mask = torch.cat([mask_prev, mask_last], dim = 1)
    return mask.float()

  @staticmethod
  def get_score_mean(normalized_score_distribution: torch.Tensor) -> torch.Tensor:
    score_center = (ZhuGoValueHead.SCORE_BUCKETS[:, 0] + ZhuGoValueHead.SCORE_BUCKETS[:, 1]) / 2
    return torch.sum(normalized_score_distribution * score_center, dim=-1, keepdim=True)
