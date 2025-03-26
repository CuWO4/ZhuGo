import go.gotypes as gotypes
import go.goboard as goboard
from agent.base import Agent
from ui.base import UI

import time

def start_game(agent1: Agent, agent2: Agent, UIClass: type, game_settings: dict):
  game = goboard.GameState.new_game(**game_settings)
  agents: dict[gotypes.Player: Agent] = {
    gotypes.Player.black: agent1,
    gotypes.Player.white: agent2
  }

  ui: UI = UIClass(*game.board.size)
  ui.update(game)

  agent1.link_to_ui(ui)
  agent2.link_to_ui(ui)

  while not game.is_over():
    move = agents[game.next_player].select_move(game)
    game = game.apply_move(move)
    ui.update(game)
    time.sleep(0.05)
