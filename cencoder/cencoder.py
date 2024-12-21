from .encodermodule import encode
from go.goboard import GameState
from go.gotypes import Player

import torch

def c_encode(game: GameState) -> torch.Tensor:
  assert game.board.num_rows <= 19 and game.board.num_cols <= 19
  
  c_board = game.board.c_board._c_board
  if game.previous_state is not None:
    c_last_board = game.previous_state.board.c_board._c_board
  else:
    c_last_board = c_board # last = current would never trigger ko
    
  c_player = 1 if game.next_player == Player.black else 2

  np_data = encode(c_board, c_last_board, c_player)
  tensor = torch.from_numpy(np_data)
  return tensor