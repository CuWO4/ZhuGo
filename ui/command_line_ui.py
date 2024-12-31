from go.goboard import GameState, Move
from go.gotypes import Point
from utils.mcts_data import MCTSData
from .base import UI

import threading
import queue
import os
import platform

def clear_console():
  if platform.system() == "Windows":
    os.system("cls")
  else:
    os.system("clear")

class CommandLineUI(UI):
  def __init__(self, row_n: int = 19, col_n: int = 19):
    super().__init__(row_n, col_n)

    self.move_queue = queue.Queue()
    
    self.listener = threading.Thread(target=self.listen_input)
    self.listener.start()

  def update(self, game_state: GameState):
    clear_console()
    print(f'{game_state.board}')

  def display_mcts(self, data: MCTSData):
    pass
  
  def get_move(self, block: bool = True) -> Move | None:
    if not block:
      return self.move_queue.get() if not self.move_queue.empty() else None
    else:
      while self.move_queue.empty(): pass
      return self.move_queue.get()

  def listen_input(self):
    while True:
      self.parse_input(input())
      
  def parse_input(self, command: str):
    command = command.split(' ')
    
    if len(command) == 1:
      move_dict = {
        'p': Move.pass_turn(),
        'r': Move.resign(),
        'u': Move.undo()
      }
      
      move = move_dict.get(command[0])
      if move is not None:
        self.move_queue.put(move)
        return
      
    if len(command) == 2:
      col_dict = { chr(i + ord('A')) : i for i in range(self.col_n) }
      row_dict = { str(i + 1) : i for i in range(self.row_n) }  

      col = col_dict.get(command[0])
      row = row_dict.get(command[1])
      
      if row is not None and col is not None:
        self.move_queue.put(Move.play(Point(row + 1, col + 1)))
        return

    print(f"unknown command {' '.join(command)}")
    print("use `A 1` .e.g to play, `p` to pass turn, `r` to resign and `u` to undo")
