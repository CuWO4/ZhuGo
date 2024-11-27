import go.gotypes as gotypes
import go.goboard as goboard
from agent.base import Agent
from ui.base import UI

import time
import multiprocessing

def start_game(agent1: Agent, agent2: Agent, UIClass: type, game_settings: dict):
  game = goboard.GameState.new_game(**game_settings)
  agents: dict[gotypes.Player: Agent] = {
    gotypes.Player.black: agent1,
    gotypes.Player.white: agent2
  }

  move_queue = multiprocessing.Queue()
  mcts_queue = multiprocessing.Queue()

  for agent in (agent1, agent2):
    agent.subscribe_move_queue(move_queue)
    agent.subscribe_mcts_queue(mcts_queue)

  ui: UI = UIClass(move_queue, mcts_queue, *game.board.size)
  ui.update(game)

  while not game.is_over():
    move = agents[game.next_player].select_move(game)
    game = game.apply_move(move)
    ui.update(game)
    time.sleep(0.05)
