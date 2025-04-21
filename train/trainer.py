from ai.manager import ModelManager
from .dataloader import BGTFDataLoader
from .exp_pool import ExpPool, Record

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import time
from datetime import datetime

__all__ = [
  'Trainer'
]


def ctrl_c_catcher(func: Callable, exit_func: Callable):
  try:
    func()
  except KeyboardInterrupt as e:
    pass
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
    batch_size: int,
    dataloader: BGTFDataLoader,
    exp_pool: ExpPool,
    batch_per_refuel: int,
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
    self.batch_size: int = batch_size
    self.dataloader: BGTFDataLoader = dataloader
    self.exp_pool: ExpPool | None = exp_pool
    self.batch_per_refuel: int = batch_per_refuel
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

      def train_body():
        optimizer = optim.SGD(
          model.parameters(),
          lr = self.base_lr,
          weight_decay = self.weight_decay,
          momentum = self.momentum
        )
        schedular = optim.lr_scheduler.CosineAnnealingLR(
          optimizer, T_max = self.T_max, eta_min = self.eta_min
        )

        def train_f(
          inputs: torch.Tensor,
          policy_targets: torch.Tensor,
          value_targets: torch.Tensor,
          ori_losses: list[float]
        ):
          policy_logits, value_logits = model(inputs)

          policy_losses = self.policy_lost_fn(policy_targets, policy_logits)
          value_losses = self.value_lost_fn(value_targets, nn.functional.tanh(value_logits))
          losses = self.policy_loss_weight * policy_losses + self.value_loss_weight * value_losses

          loss = torch.mean(losses)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          original_loss = sum(ori_losses) / len(ori_losses)

          schedular.step()

          print(
            f'{"train:":>10}'
            f'{self.exp_pool.size:12}'
            f'{loss.item():12.3f}'
            f'{original_loss:12.3f}'
            f'{self.exp_pool.loss_mean:12.3f}'
            f'{"press Ctrl-C to stop":>30}'
          )

          writer.add_scalar('train/batch-loss', loss, meta.batches)
          writer.add_scalar('train/pool-loss', self.exp_pool.loss_mean, meta.batches)
          writer.add_scalar('train/lr', schedular.get_last_lr()[0], meta.batches)

          return losses
        # train_f end

        last_checkpoint_time = time.time()

        for records in self.dataloader:
          print('refueling...')
          self.exp_pool.insert_record(records)

          for _ in range(self.batch_per_refuel):
            self.exp_pool.train_on_batch(self.batch_size, train_f, device = device)
            meta.batches += 1

          if self.test_dataloader is not None:
            test_records = next(iter(self.test_dataloader))
            inputs, policies, values, _, _ = Record.stack(test_records, device = device)

            with torch.no_grad():
              model.eval()
              policy_logits, value_logits = model(inputs)
              model.train()

            policy_losses = self.policy_lost_fn(policies, policy_logits)
            value_losses = self.value_lost_fn(values, nn.functional.tanh(value_logits))
            losses = self.policy_loss_weight * policy_losses + self.value_loss_weight * value_losses
            loss = torch.mean(losses)

            print(
              f'{"test:":>10}'
              f'{loss.item():12.3f}'
            )
            writer.add_scalar('test/loss', loss, meta.batches)

          if time.time() - last_checkpoint_time >= self.checkpoint_interval_sec:
            print('saving checkpoint...')
            self.model_manager.save_checkpoint(model)
            for name, param in model.named_parameters():
              writer.add_histogram(f'weights/{name}', param, meta.batches)
              if param.grad is not None:
                writer.add_histogram(f'grads/{name}', param.grad, meta.batches)
            print(f'checkpoint saved at {datetime.now().strftime("%H:%M:%S")}')
            last_checkpoint_time = time.time()

      # train_body end

      def stop_handling():
        print('stopped. saving...')
        self.model_manager.save_model(model)
        self.model_manager.save_meta(meta)

      ctrl_c_catcher(train_body, stop_handling)
