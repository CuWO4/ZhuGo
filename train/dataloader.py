from dataloader.bgtf_zstd_loader import load_file
from .exp_pool import Record
from ai.encoder.base import Encoder
from ai.encoder.zhugo_encoder import ZhuGoEncoder
from ai.utils.rotate import get_d8_sym_tensors

import torch
import torch.multiprocessing as mp
import os
import random
import time
from itertools import cycle
from typing import Iterator, Callable

__all__ = [
  'BGTFDataLoader',
]


def get_d8_sym_records(
  input: torch.Tensor,
  policy_target: torch.Tensor,
  value_target: torch.Tensor
) -> Iterator[Record]:
  '''input(C, 19, 19), policy_target(19 * 19 + 1), value_target(1)'''
  assert (
    input.shape == (input.shape[0], 19, 19)
    and policy_target.shape == (362,)
    and value_target.shape == (1,)
  ), (
    f"<{get_d8_sym_records.__name__}>: invalid tensor shape "
    f"{input.shape=} {policy_target.shape=} {value_target.shape=}"
  )

  return (
    Record(
      input = new_input.unsqueeze(0),
      policy_target = torch.cat((new_policy.reshape(361), policy_target[361:]), dim = 0).unsqueeze(0),
      value_target = value_target.clone().unsqueeze(0),
      loss = float('inf'),
    )
    for new_input, new_policy in zip(
      get_d8_sym_tensors(input),
      get_d8_sym_tensors(policy_target[:361].view(19, 19)),
    )
  )

def busy_wait(end_cond: Callable, interval_ms: int):
  while not end_cond(): time.sleep(interval_ms * 1e-3)

# the bgtf is encoded time dependently, and is used to refuel exp pool
# so pytorch architecture does not suite. though extending
# torch.utils.data.IterableDataset can also implement the function,
# unnecessary dependencies are imported. a from zero implementation
# would be more concise.
class BGTFDataLoader:
  def __init__(
    self,
    root: str, # root directory of data
    batch_size: int,
    *,
    prefetch_batch: int = 2,
    debug: bool = False
  ):
    super().__init__()
    self.debug = debug
    self.cached = mp.Queue()
    mp.Process(
      target = self.decode_worker,
      args = (
        self.cached,
        root,
        batch_size,
        prefetch_batch,
        debug
      ),
      daemon = True,
    ).start()

  def __iter__(self) -> Iterator:
    while True:
      yield self.cached.get(block = True)

  @staticmethod
  def record_stream(root: str, encoder: Encoder, debug: bool) -> Iterator[Record]:
    files = [path for f in os.listdir(root) if os.path.isfile(path := os.path.join(root, f))]
    random.shuffle(files)
    for file in cycle(files):
      if debug:
        print(f'<record_stream> loading {file}')
      try:
        records = [
          record
          for game, policy, value in load_file(file)
          for record in get_d8_sym_records(encoder.encode(game), policy, value)
        ]
        random.shuffle(records)
        yield from records
      except Exception as e:
        print(f'<record_stream> error `{e}` happened when handling `{file}`')

  @staticmethod
  def decode_worker(
    result_queue: mp.Queue,
    root: str,
    batch_size: int,
    prefetch_batch: int,
    debug: bool,
  ) -> None:
    random.seed()
    encoder = ZhuGoEncoder(device = 'cpu')
    batch = []
    for record in BGTFDataLoader.record_stream(root, encoder, debug):
      batch.append(record.share_memory_())
      if len(batch) >= batch_size:
        busy_wait(lambda: result_queue.qsize() < prefetch_batch, 50)
        result_queue.put(batch)
        batch = []
