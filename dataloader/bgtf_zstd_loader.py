from .bgtf_loader import load_bytes
from go.goboard import GameState

import zstandard as zstd
from typing import Iterator
import torch

dctx = zstd.ZstdDecompressor()

def load_file(path: str) -> Iterator[tuple[GameState, torch.Tensor, torch.Tensor]]:
  with open(path, 'rb') as f:
    yield from load_bytes(dctx.decompress(f.read()))
