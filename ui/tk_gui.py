import ui.base as base

from ui.utils import MCTSData
from go.goboard import GameState, Move, Point
from go.gotypes import Player
from utils.move_idx_transformer import idx_to_move

import tkinter as tk
import multiprocessing

__all__ = [
  'TkGUI'
]

class TkGUI(base.UI):
  def __init__(self,
               move_queue: multiprocessing.Queue,
               mcts_queue: multiprocessing.Queue = None,
               row_n: int = 19, col_n: int = 19):
    assert row_n == col_n

    super().__init__(move_queue, mcts_queue, row_n, col_n)
    self.game_state_queue = multiprocessing.Queue()

    self.renderer_process = multiprocessing.Process(target=TkRenderer, args=(self.game_state_queue, move_queue, mcts_queue, row_n))
    self.renderer_process.start()

  def update(self, game_state: GameState):
    self.game_state_queue.put(game_state)

class TkRenderer:
  def __init__(self,
               game_state_queue: multiprocessing.Queue,
               move_queue: multiprocessing.Queue,
               mcts_queue: multiprocessing.Queue,
               board_size: int,
               cell_size: int=40,
               padding: int=60):
    self.game_state_queue: multiprocessing.Queue = game_state_queue
    self.move_queue: multiprocessing.Queue = move_queue
    self.mcts_queue: multiprocessing.Queue | None = mcts_queue
    self.cur_game_state: GameState | None = None
    self.cur_mcts_data: MCTSData | None = None
    self.refresh_ms: int = 100

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
    self.canvas.bind("<Leave>", self.on_mouse_leave)

    def check_queue():
      while not self.game_state_queue.empty():
        self.cur_game_state = self.game_state_queue.get()

      while self.mcts_queue is not None and not self.mcts_queue.empty():
        self.cur_mcts_data = self.mcts_queue.get()
        
      self.draw_board()

      self.root.after(self.refresh_ms, check_queue)

    self.root.after(self.refresh_ms, check_queue)
    self.root.mainloop()

  def draw_board(self):
    if self.cur_game_state is None:
      return

    self.canvas.delete("all")

    self.draw_lines()

    self.draw_star_points()

    self.draw_hover()

    self.draw_pieces()

    self.draw_mcts_state()

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

  def draw_piece(self, x, y, color, expand=0):
    cx = self.padding + x * self.cell_size
    cy = self.padding + y * self.cell_size
    radius = self.cell_size // 2 - 2 + expand
    self.canvas.create_oval(
      cx - radius, cy - radius, cx + radius, cy + radius,
      fill=color, outline=color
    )

  def draw_mcts_state(self):
    if self.cur_mcts_data is None:
      return
    
    best_move_pos = self.cur_mcts_data.best_pos()
    if best_move_pos is not None:
      best_move_y, best_move_x = best_move_pos
      self.draw_piece(best_move_x, best_move_y, '#186A3B', expand=3)
    
    for x in range(self.board_size):
      for y in range(self.board_size):
        q, visited_time = self.cur_mcts_data.get(row=y, col=x)
        if visited_time < 5:
          continue
        color_table = [
          (0.1,  '#A93226'), # dark red
          (0.3,  '#E74C3C'), # light red
          (0.45, '#EC7063'), # lighter red
          (0.55, '#F4D03F'), # yellow
          (0.7,  '#82E0AA'), # lighter green
          (0.9,  '#58D68D'), # light green
          (1,    '#239B56')  # dark green
        ]
        candidate_color = color_table[0][1]
        for upper_bound, color in color_table:
           if q <= upper_bound:
             candidate_color = color
             break
        cx = self.padding + x * self.cell_size
        cy = self.padding + y * self.cell_size
        vertical_bias = 5
        self.draw_piece(x, y, candidate_color)
        self.canvas.create_text(cx, cy - vertical_bias, text=f'{q:.2f}', font=('Consolas', 10), fill='black')
        self.canvas.create_text(cx, cy + vertical_bias, text=f'{int(visited_time)}', font=('Consolas', 10), fill='black')

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
      
  def on_mouse_leave(self, event):
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

