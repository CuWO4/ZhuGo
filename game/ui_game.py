import go.gotypes as gotypes
import go.goboard as goboard
from agent.base import Agent

import time
import multiprocessing

def start_game(agent1: Agent, agent2: Agent, UIClass: type, board_size: int = 19):
  game = goboard.GameState.new_game(board_size)
  agents: dict[gotypes.Player: Agent] = {
    gotypes.Player.black: agent1,
    gotypes.Player.white: agent2
  }

  move_queue = multiprocessing.Queue()

  for agent in (agent1, agent2):
    if hasattr(agent, 'subscribe_move_queue'):
      agent.subscribe_move_queue(move_queue)

  ui = UIClass(move_queue)
  ui.update(game)
  
  while not game.is_over():
    move = agents[game.next_player].select_move(game)
    game = game.apply_move(move)
    ui.update(game)
    time.sleep(0.05)
    
  # TODO: let the master and slave threads terminated at the same time
