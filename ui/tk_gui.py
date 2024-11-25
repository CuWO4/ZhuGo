import ui.base as base

from go.goboard import GameState, Move, Point
from go.gotypes import Player

import tkinter as tk
import multiprocessing

__all__ = [
  'TkGUI'
]

class TkGUI(base.UI):
  def __init__(self, move_queue: multiprocessing.Queue, row_n: int = 19, col_n: int = 19):
    assert row_n == col_n

    super().__init__(move_queue, row_n, col_n)
    self.game_state_queue = multiprocessing.Queue()
    
    self.renderer_process = multiprocessing.Process(target=TkRenderer, args=(self.game_state_queue, move_queue, row_n))
    self.renderer_process.start()
    
  def update(self, game_state: GameState):
    self.game_state_queue.put(game_state)
  
class TkRenderer:
  def __init__(self, game_state_queue: multiprocessing.Queue, move_queue: multiprocessing.Queue, board_size: int, cell_size: int=40, padding: int=60):
    super().__init__()
    self.game_state_queue: multiprocessing.Queue = game_state_queue
    self.move_queue: multiprocessing.Queue = move_queue
    self.cur_game_state: GameState | None = None
    
    self.board_size: int = board_size
    self.cell_size: int = cell_size
    self.padding: int = padding
    self.windows_size: int = cell_size * (board_size - 1) + 2 * padding
    self.root: tk.Tk = tk.Tk()
    self.root.title("ZhuGo")
    self.canvas: tk.Canvas = tk.Canvas(
      self.root,
      width=self.windows_size,
      height=self.windows_size,
      bd=0,
      highlightthickness=0,
      bg="#f5b041"
    )
    self.canvas.pack()
    
    self.mouse_hover_pos: tuple[int] | None = None

    # Add buttons for Pass and Resign
    self.pass_turn_button: tk.Button = tk.Button(self.root, text="Pass", command=self.on_pass_turn)
    self.pass_turn_button.pack(side=tk.LEFT, padx=10, pady=10)

    self.resign_button: tk.Button = tk.Button(self.root, text="Resign", command=self.on_resign)
    self.resign_button.pack(side=tk.LEFT, padx=10, pady=10)

    self.canvas.bind("<Button-1>", self.on_click)
    self.canvas.bind("<Motion>", self.on_mouse_move)
    
    def check_queue():
      while not self.game_state_queue.empty():
        self.cur_game_state = self.game_state_queue.get()
      
      self.draw_board()
        
      self.root.after(10, check_queue)

    self.root.after(10, check_queue)
    self.root.mainloop()
    
  def draw_board(self):
    if self.cur_game_state is None:
      return
    
    self.canvas.delete("all")

    self.draw_lines()

    self.draw_star_points()

    self.draw_hover()
    
    self.draw_pieces()
    
    self.draw_message()
          
  def draw_lines(self):
    for x in range(self.board_size):
      # horizontal lines
      self.canvas.create_line(
        self.padding,
        self.padding + x * self.cell_size,
        self.padding + (self.board_size - 1) * self.cell_size,
        self.padding + x * self.cell_size,
        fill="black"
      )
      # vertical lines
      self.canvas.create_line(
        self.padding + x * self.cell_size,
        self.padding,
        self.padding + x * self.cell_size,
        self.padding + (self.board_size - 1) * self.cell_size,
        fill="black"
      )

  def draw_star_points(self):
    star_positions = self.get_star_positions()
    for x, y in star_positions:
      cx = self.padding + x * self.cell_size
      cy = self.padding + y * self.cell_size
      radius = 3
      self.canvas.create_oval(
        cx - radius, cy - radius, cx + radius, cy + radius,
        fill="black"
      )

  def get_star_positions(self):
    if self.board_size == 19:
      points = [3, 9, 15]
    elif self.board_size == 13:
      points = [3, 6, 9]
    elif self.board_size == 9:
      points = [2, 4, 6]
    else:
      return []  # non-standard

    return [(x, y) for x in points for y in points]
  
  def draw_hover(self):
    if self.mouse_hover_pos is None:
      return

    cx = self.padding + self.mouse_hover_pos[0] * self.cell_size
    cy = self.padding + self.mouse_hover_pos[1] * self.cell_size
    radius = self.cell_size // 2
    color = "#58d68d"
    self.canvas.create_oval(
      cx - radius, cy - radius, cx + radius, cy + radius,
      fill=color, outline=color, width=5
    )
    
  def draw_pieces(self):
    for x in range(self.board_size):
      for y in range(self.board_size):
        stone = self.cur_game_state.board.get(Point(row = y + 1, col= x + 1))
        if stone == Player.black:
          self.draw_piece(x, y, "black")
        elif stone == Player.white:
          self.draw_piece(x, y, "white")

  def draw_piece(self, x, y, color):
    cx = self.padding + x * self.cell_size
    cy = self.padding + y * self.cell_size
    radius = self.cell_size // 2 - 2
    self.canvas.create_oval(
      cx - radius, cy - radius, cx + radius, cy + radius,
      fill=color, outline=color
    )
    
  def draw_message(self):
    if self.cur_game_state.is_over():
      winner = 'white' if self.cur_game_state.winner() == Player.white else 'black'
      game_result = self.cur_game_state.game_result()
      message = f'{winner} wins  {game_result}'
      self.canvas.create_text(
        self.windows_size / 2, self.windows_size / 2, 
        text=message, font=("Arial", 60), fill="red"
      )
    
  def on_mouse_move(self, event: tk.Event):
    x = round((event.x - self.padding) / self.cell_size)
    y = round((event.y - self.padding) / self.cell_size)
    if 0 <= x < self.board_size and 0 <= y < self.board_size:
      if self.mouse_hover_pos != (x, y):
        self.mouse_hover_pos = (x, y)
        self.draw_board()
    else:
      self.mouse_hover_pos = None

  def on_click(self, event: tk.Event):
    x = round((event.x - self.padding) / self.cell_size)
    y = round((event.y - self.padding) / self.cell_size)
    if 0 <= x < self.board_size and 0 <= y < self.board_size:
      move = Move.play(Point(row = y + 1, col = x + 1))
      base.UI.enqueue_move(self.move_queue, move)

  def on_pass_turn(self):
    move = Move.pass_turn()
    base.UI.enqueue_move(self.move_queue, move)

  def on_resign(self):
    move = Move.resign()
    base.UI.enqueue_move(self.move_queue, move)

