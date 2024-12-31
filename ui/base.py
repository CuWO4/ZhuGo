from go.goboard import Move, GameState
from utils.mcts_data import MCTSData

__all__ = [
  'UI'
]

class UI:
  def __init__(self, row_n: int = 19, col_n: int = 19):
    self.row_n: int = row_n
    self.col_n: int = col_n
  
  def update(self, game_state: GameState):
    raise NotImplementedError()
  
  def display_mcts(self, data: MCTSData):
    raise NotImplementedError()
  
  def get_move(self, block: bool = True) -> Move | None:
    raise NotImplementedError()
