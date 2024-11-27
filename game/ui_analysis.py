import go.gotypes as gotypes
import go.goboard as goboard
from agent.base import Agent
from ui.base import UI

import time
import multiprocessing

def start_analysis(agent: Agent, UIClass: type, board_size: int = 19, komi: float = 7.5):
  game = goboard.GameState.new_game(board_size, komi)

  move_queue = multiprocessing.Queue()
  mcts_queue = multiprocessing.Queue()

  agent.subscribe_move_queue(move_queue)
  agent.subscribe_mcts_queue(mcts_queue)

  ui: UI = UIClass(move_queue, mcts_queue, board_size, board_size)
  ui.update(game)
  
  while not game.is_over():
    move = agent.select_move(game)
    game = game.apply_move(move)
    ui.update(game)
    time.sleep(0.05)
    