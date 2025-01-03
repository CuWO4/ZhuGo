from agent.base import Agent
from go.goboard import GameState, Move

__all__ = [
  'HumanAgent'
]

class HumanAgent(Agent):
  def __init__(self):
    super().__init__()
    
  def select_move(self, game_state: GameState) -> Move:
    assert self.ui is not None
    
    return self.ui.get_move(block=True)
