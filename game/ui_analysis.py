import go.goboard as goboard
from agent.base import Agent
from ui.base import UI

import time

def start_analysis(agent: Agent, UIClass: type, game_settings: dict):
  game = goboard.GameState.new_game(**game_settings)

  ui: UI = UIClass(*game.board.size)
  ui.update(game)
  
  agent.link_to_ui(ui)
  
  while not game.is_over():
    move = agent.select_move(game)
    game = game.apply_move(move)
    ui.update(game)
    time.sleep(0.05)
    
  input(f'game ends. winner: {game.winner()}. press Enter to continue...')
