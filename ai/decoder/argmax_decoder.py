from .base import Decoder

from go.goboard import Move
from go.gotypes import Point
from utils.move_idx_transformer import idx_to_move

import torch

__all__ = [
  'ArgmaxDecoder'
]

class ArgmaxDecoder(Decoder):
  def __init__(self, size: tuple[int, int], *, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
    super().__init__(size, device=device)

  def decoder(self, policy_tensor: torch.Tensor) -> Move:
    policy_tensor = policy_tensor.to(device=self.device)

    assert policy_tensor.size() == self.size

    row, col = torch.unravel_index(torch.argmax(policy_tensor), policy_tensor.size())
    row, col = row.item(), col.item()

    return Move.play(Point(row=row, col=col))
