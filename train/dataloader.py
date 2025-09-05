from dataloader.bgtf_zstd_loader import load_file
from ai.encoder.base import Encoder
from ai.utils.rotate import get_d8_sym_tensors

import torch
import torch.multiprocessing as mp
import os
import random
from itertools import cycle
from operator import itemgetter
from typing import Iterator, Callable

__all__ = [
  'BGTFDataLoader',
]


class BGTFDataLoader:
  def __init__(
    self,
    root: str, # root directory of data
    batch_size: int,
    encoder: Encoder,
    *,
    prefetch_batch: int = 300,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    # file decoding, data enhancement and move to share memory are
    # relevantly slow, requires big prefetch batch
    debug: bool = False
  ):
    super().__init__()
    ctx = mp.get_context('spawn') # fork mode introduce instability
    self.device = device
    self.debug = debug
    self.cached = ctx.Queue(maxsize = prefetch_batch)

    encoder.device = 'cpu' # store prefetched tensors on cpu

    self.worker = ctx.Process(
      target = self.decode_worker,
      kwargs = {
        "result_queue": self.cached,
        "root": root,
        "batch_size": batch_size,
        "encoder": encoder,
        "debug": debug,
      },
    )
    self.worker.start()

  def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    while True:
      inputs, policies, win_rate, ownership, score = self.cached.get(block = True)
      inputs = inputs.to(device = self.device)
      policies = policies.to(device = self.device)
      win_rate = win_rate.to(device = self.device)
      ownership = ownership.to(device = self.device)
      score = score.to(device = self.device)
      yield inputs, policies, win_rate, ownership, score

  def close(self):
    self.worker.terminate()

  @staticmethod
  def input_target_stream(root: str, encoder: Encoder, debug: bool) \
    -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    files = [path for f in os.listdir(root) if os.path.isfile(path := os.path.join(root, f))]
    random.shuffle(files)
    for file in cycle(files):
      if debug:
        print(f'<record_stream> loading {file}')

      results = []
      for game, policy, win_rate, ownership, score in load_file(file):
        rotated_tensors = [
          (
            rotated_input.cpu().unsqueeze_(0),
            torch.cat(
              (rotated_policy.reshape(361), policy[361:])
            ).cpu().unsqueeze_(0),
            win_rate.cpu().unsqueeze(0),
            rotated_ownership.cpu().reshape(361).unsqueeze_(0),
            score.cpu().unsqueeze(0),
          )
          for rotated_input, rotated_policy, rotated_ownership in zip(
            get_d8_sym_tensors(encoder.encode(game)),
            get_d8_sym_tensors(policy[:361].view(19, 19)),
            get_d8_sym_tensors(ownership.view(19, 19)),
          )
        ]
        random.shuffle(rotated_tensors)
        results += rotated_tensors[:2] # to avoid overfitting and wasting

      random.shuffle(results)
      yield from results

  @staticmethod
  def decode_worker(
    result_queue: mp.Queue,
    root: str,
    batch_size: int,
    encoder: Encoder,
    debug: bool,
  ) -> None:
    random.seed()
    batch = []
    try:
      for tensors in BGTFDataLoader.input_target_stream(root, encoder, debug):
        batch.append(tensors)
        if len(batch) >= batch_size:
          batch_tensor = (
            torch.cat(list(map(itemgetter(0), batch)), dim = 0),
            torch.cat(list(map(itemgetter(1), batch)), dim = 0),
            torch.cat(list(map(itemgetter(2), batch)), dim = 0),
            torch.cat(list(map(itemgetter(3), batch)), dim = 0),
            torch.cat(list(map(itemgetter(4), batch)), dim = 0),
          )
          result_queue.put(batch_tensor, block = True)
          batch = []
    except KeyboardInterrupt:
      return
    except Exception:
      import traceback, sys
      traceback.print_exc()
      sys.stderr.flush()
      raise
