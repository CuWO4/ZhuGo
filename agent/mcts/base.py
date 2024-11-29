from go.goboard import GameState, Move
from utils.move_idx_transformer import move_to_idx
from utils.eye_identifier import is_point_an_eye

import numpy as np
from typing import TypeVar
import multiprocessing as mp

__all__ = [
  'Node'
]

T = TypeVar('T', bound='Node')
class Node:
  def __init__(self, *, game_state: GameState, pool: mp.Pool, c: float):
    '''
    indexings are consistent with utils.move_idx_transformer.py
    
    Arguments
    --------------
    game_state
      game state the Node represents
      
    pool
      a Agent-managed threads pool
      
    c 
      exploring temperature; higher c strategy would prefer exploring
      underexplored situation, while lower c strategy would prefer
      exploring currently more favorable situation to increase its
      confidence

    all Node subclasses need these three arguments
      
    Attributes
    --------------
    q
      value estimation of each move

    visited_times
      visited times of each move
      
    game_state
      game state of Node

    c
      exploring temperature

    row_n, col_n
      board size

    policy_size
      row_n * col_n + 1, including pass turn

    branches
      a policy_size-long list, Nodes for subsituations, None if not 
      explored
      
    branch
      certain branch of certain move

    legal_mask, legal_play_count
      legal plays (which do not include pass turn and resign) according 
      to go rule, filling own eye is illegal
      
      legal moves are set 0, otherwise a negative number with 
      a large absolute value. add legal_mask to ucb to avoid illegal
      moves are searched or applied
    '''
    self.game_state: GameState = game_state
    self.pool = pool
    self.c = c
    
    self.row_n: int = game_state.board.num_rows
    self.col_n: int = game_state.board.num_cols
    self.policy_size: int = self.row_n * self.col_n + 1

    self.branches: list[Node] = [None] * self.policy_size

    self.legal_mask = np.full(self.policy_size, -1e5, dtype=np.float64)
    self.legal_play_count = 0
    for move in game_state.legal_moves():
      if move.is_pass:
        continue
      if not move.is_resign and \
        not is_point_an_eye(self.game_state.board, move.point, game_state.next_player):
        idx = move_to_idx(move, self.game_state.board.size)
        self.legal_mask[idx] = 0
        self.legal_play_count += 1

  def propagate(self) -> object:
    '''the return value type is determined by the Node subclass
    and is only processed when the propagate method is called recursively
    by subclass. External callers do not need to care about the return value.
    '''
    raise NotImplementedError()
  
  def branch(self: T, move: Move) -> T:
    '''return the Node representing state applied with certain move'''
    raise NotImplementedError()


  @property
  def q(self) -> np.ndarray[float]:
    '''torch.tensor can be easily transfer to numpy without great cost'''
    raise NotImplementedError()
  
  @property
  def visited_times(self) -> np.ndarray[int]:
    raise NotImplementedError()
