import torch

def get_d8_sym_tensors(tensor: torch.Tensor) -> list[torch.Tensor]:
  assert len(tensor.shape) >= 2

  return [
    torch.rot90(t, k, dims=(-2, -1))
    for k in range(4)
    for t in [tensor, torch.flip(tensor, dims=(-1,))]
  ]
