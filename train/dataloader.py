from dataloader.bgtf_zstd_loader import load_file
from .exp_pool import Record
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
class BGTFDataLoader(Iterator):
  def __init__(
    self,
    root: str, # root directory of data
    batch_size: int,
    debug: bool = False
  ):
    super().__init__()

    min_record_num = 3 * batch_size
    max_record_num = 10 * batch_size

    self.batch_size = batch_size
    self.debug = debug

    self.cached = mp.Queue(maxsize = max_record_num)
    self.put_permission = mp.Event()
    self.put_permission.set()
    # disable subprocess putting new record when getting next batch,
    # avoiding racing. slightly improve performance.

    decoder_process = mp.Process(
      target = self.decode_worker,
      args = (
        self.cached,
        root,
        min_record_num,
        self.put_permission,
        debug
      ),
      daemon = True,
    )
    decoder_process.start()

  def __iter__(self) -> Iterator:
    return self

  def __next__(self) -> list[Record]:
    busy_wait(lambda: self.cached.qsize() >= self.batch_size, 1)
    self.put_permission.clear()
    records = [self.cached.get(block = True) for _ in range(self.batch_size)]
    self.put_permission.set()
    return records

  @staticmethod
  def decode_worker(
    result_queue: mp.Queue,
    root: str,
    min_record_num: int,
    put_permission: mp.Event,
    debug: bool,
  ) -> None:
    files = [path for f in os.listdir(root) if os.path.isfile(path := os.path.join(root, f))]
    random.seed()
    random.shuffle(files)

    encoder = ZhuGoEncoder(device = 'cpu')

    for file in cycle(files):
      busy_wait(lambda: result_queue.qsize() < min_record_num, 50)

      if debug:
        print(f'<{__file__}> loading {file}')

      try:
        records = [
          record
          for game, policy, value in load_file(file)
          for record in get_d8_sym_records(encoder.encode(game), policy, value)
        ]
        random.shuffle(records)
        for record in records:
          put_permission.wait()
          result_queue.put(record)
      except Exception as e:
        print(f'<{__file__}> error processing `{file}`: {e}')
