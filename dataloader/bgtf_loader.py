from go.goboard import Move, GameState
from go.gotypes import Point, Player
import go.scoring as scoring

from struct import unpack
import torch
import io
from io import BufferedReader
from typing import Iterator

__all__ = [
  'load_file',
  'load_bytes',
]

def load_turn(file: BufferedReader, endian: str) -> tuple[Move | None, tuple[float]]:
  '''@return move, policy_target'''
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

  return move, data[2:]

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

  records = []

  for _ in range(turn_count):
    move, policy_target = load_turn(file, endian)

    if __debug__:
      if move and move.is_play and game.board.get(point=move.point) is not None:
        print(f'runtime warning: bad move on offset <{file.tell():x}>')

        print(game.board, game.next_player, move.point)

        print(game.apply_move(move).board)

    if move is not None:
      game = game.apply_move(move)

    policy_tensor = torch.tensor(policy_target, device = 'cpu')
    value_tensor = torch.tensor([value_target], device = 'cpu')

    records.append((game, policy_tensor, value_tensor))

    value_target = -value_target

  # analyze ownership & score
  endgame = records[-1][0]
  territory = scoring.evaluate_territory(endgame.board)

  ownership_tensor = torch.zeros(endgame.board.num_rows * endgame.board.num_cols, device='cpu')
  for point, status in territory.map.items():
    if status == Player.black or status == 'territory_b':
      ownership_tensor[(point.row - 1) * endgame.board.num_cols + point.col - 1] = 1
    elif status == Player.white or status == 'territory_w':
      ownership_tensor[(point.row - 1) * endgame.board.num_cols + point.col - 1] = -1

  score = territory.num_black_territory + territory.num_black_stones \
    - (territory.num_white_territory + territory.num_white_stones) \
    - endgame.komi
  score_tensor = torch.tensor([score], device='cpu')

  for game, policy_tensor, value_tensor in records:
    yield game, policy_tensor, value_tensor, ownership_tensor, score_tensor
    ownership_tensor = -ownership_tensor
    score_tensor = -score_tensor

def load_reader(file: BufferedReader) -> Iterator[tuple]:
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

  for game_offset in game_offsets:
    yield from load_game(file, game_offset, endian)

def load_file(path: str) -> Iterator[tuple[GameState, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
  '''return list(game_state, policy_target(N * M + 1), value_target(1), ownership_target(N * M + 1), score(1))'''
  with open(path, 'rb') as f:
    yield from load_reader(f)

def load_bytes(data: bytes) -> Iterator[tuple[GameState, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
  '''return list(game_state, policy_target(N * M + 1), value_target(1), ownership_target(N * M + 1), score(1))'''
  bytes_io = io.BytesIO(data)
  buffered_reader = io.BufferedReader(bytes_io)
  yield from load_reader(buffered_reader)
