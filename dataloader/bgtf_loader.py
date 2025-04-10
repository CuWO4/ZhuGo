from go.goboard import Move, GameState
from go.gotypes import Point

from struct import unpack
import torch
from io import BufferedReader
from typing import Iterator

__all__ = [
  'load',
]


def load_turns(
  file: BufferedReader,
  turn_count: int,
  endian: str,
) -> Iterator[tuple[Move | None, tuple[float]]]:
  '''return last move, target policy distribution (row first encoding, 361 + 1)'''
  for _ in range(turn_count):
    data = unpack(f'{endian}2H362f', file.read(2 * 2 + 362 * 4))

    row, col = data[0 : 2]
    if 0 <= row < 19 and 0 <= col < 19:
      move = Move.play(Point(row + 1, col + 1))
    elif row == col == 19: # new game, no move
      move = None
    elif row == col == 20:
      move = Move.pass_turn()
    else:
      move = Move.resign()
      print(f'runtime warning: an assign movement decoded at offset <{file.tell():x}>')

    yield move, data[2:]

def load_game(file: BufferedReader, game_offset: int, endian: str) -> Iterator[tuple]:
  file.seek(game_offset, 0) # 0 for whence = SEEK_SET

  turn_count, = unpack(f'{endian}I', file.read(4))

  winner, = unpack(f'{endian}I', file.read(4))
  if winner == 0: # black
    value_target = 1
  elif winner == 1: # white
    value_target = -1
  elif winner == 2: # draw
    value_target = 0
  else:
    raise RuntimeError(f'unknown winner {winner}')

  game = GameState.new_game()

  for move, policy_target in load_turns(file, turn_count, endian):
    if move is not None:
      game = game.apply_move(move)

    policy_target = torch.tensor(policy_target[:361], device = 'cpu').view(19, 19)
    policy_target /= torch.sum(policy_target) + 1e-8
    value_target = torch.tensor([value_target], device = 'cpu')

    yield game, policy_target, value_target

    value_target = -value_target

def load_file(file: BufferedReader) -> list[tuple]:
  magic_number, = unpack('>I', file.read(4))

  if magic_number == 0x3456789A:
    endian = '>'
  elif magic_number == 0x9A785634:
    endian = '<'
  else:
    raise RuntimeError(f'corrupted file: unknown magic number {magic_number:x}')

  file.read(4 + 64) # version (uint32) + reserved (64B)

  game_count, = unpack(f'{endian}I', file.read(4))

  game_offsets = unpack(f'{endian}{game_count}Q', file.read(8 * game_count))

  return [result for game_offset in game_offsets for result in load_game(file, game_offset, endian)]

def load(path: str) -> list[tuple[GameState, torch.Tensor, torch.Tensor]]:
  '''return list(game_state, policy_target(N, M), value_target(1))'''
  with open(path, 'rb') as f:
    return load_file(f)
