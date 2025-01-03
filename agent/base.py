from go.goboard import GameState, Move
from ui.base import UI

__all__ = [
  'Agent'
]

class Agent:
  def __init__(self):
    self.ui = None
    
  def link_to_ui(self, ui: UI):
    self.ui = ui
    
  def select_move(self, game_state: GameState) -> Move:
    raise NotImplementedError()
    