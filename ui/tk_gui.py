from .base import UI

from utils.mcts_data import MCTSData
from go.goboard import GameState, Move, Point
from go.gotypes import Player

import tkinter as tk
import multiprocessing

__all__ = [
  'TkGUI'
]

class GameStateProxy: # GameState includes py capsule object, cannot be serialized
  def __init__(self, game_state: GameState) -> None:
    self.next_player = game_state.next_player

    self.stones = [
      [game_state.board.get(Point(row = row + 1, col = col + 1)) 
       for col in range(game_state.board.num_cols)]
      for row in range(game_state.board.num_rows)
    ]

    self.is_over = game_state.is_over()
    if self.is_over:
      self.winner = game_state.winner()
      self.game_result = game_state.game_result()
    

class TkGUI(UI):
  def __init__(
    self,
    move_queue: multiprocessing.Queue,
    mcts_queue: multiprocessing.Queue = None,
    row_n: int = 19, col_n: int = 19
  ):
    super().__init__(move_queue, mcts_queue, row_n, col_n)
    self.game_state_queue = multiprocessing.Queue()

    self.renderer_process = multiprocessing.Process(
      target=TkRenderer, 
      args=(self.game_state_queue, move_queue, mcts_queue, row_n, col_n)
    )
    self.renderer_process.start()

  def update(self, game_state: GameState):
    self.game_state_queue.put(GameStateProxy(game_state))

class TkRenderer:
  def __init__(
    self,
    game_state_queue: multiprocessing.Queue,
    move_queue: multiprocessing.Queue,
    mcts_queue: multiprocessing.Queue,
    row_n: int, col_n: int,
    cell_size: int=40,
    padding: int=60,
    back_ground_color: str = '#CDAC6A',
    winning_rate_bar_height: int = 25,
    font: str = 'Consolas'
  ):
    self.game_state_queue: multiprocessing.Queue = game_state_queue
    self.move_queue: multiprocessing.Queue = move_queue
    self.mcts_queue: multiprocessing.Queue | None = mcts_queue
    self.cur_game_state: GameStateProxy | None = None
    self.cur_mcts_data: MCTSData | None = None
    self.refresh_ms: int = 100

    self.row_n: int = row_n
    self.col_n: int = col_n
    self.cell_size: int = cell_size
    self.padding: int = padding
    
    self.winning_rate_bar_height: int = winning_rate_bar_height

    self.back_ground_color = back_ground_color

    self.font: str = font
    
    self.window_w: int = cell_size * (col_n - 1) + 2 * padding
    self.window_h: int = cell_size * (row_n - 1) + 2 * padding
    self.root: tk.Tk = tk.Tk()
    self.root.title("ZhuGo")
    self.canvas: tk.Canvas = tk.Canvas(
      self.root,
      width=self.window_w,
      height=self.window_h + padding,
      bd=0,
      highlightthickness=0,
      bg=self.back_ground_color
    )
    self.canvas.pack()

    self.mouse_hover_pos: tuple[int] | None = None

    # Add buttons for Pass, Resign and Undo
    self.pass_turn_button: tk.Button = tk.Button(self.root, text="Pass", command=self.on_pass_turn_button)
    self.pass_turn_button.pack(side=tk.LEFT, padx=10, pady=10)

    self.resign_button: tk.Button = tk.Button(self.root, text="Resign", command=self.on_resign_button)
    self.resign_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    self.undo_button: tk.Button = tk.Button(self.root, text='Undo', command=self.on_undo_botton)
    self.undo_button.pack(side=tk.LEFT, padx=10, pady=10)

    self.canvas.bind("<Button-1>", self.on_click)
    self.canvas.bind("<Motion>", self.on_mouse_move)
    self.canvas.bind("<Leave>", self.on_mouse_leave)
    self.root.bind("<Control-z>", self.on_ctrl_z)
    self.root.bind("<Tab>", self.on_tab)

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

    self.draw_cur_player()

    self.draw_message()

  def draw_cur_player(self):
    color = 'black' if self.cur_game_state.next_player == Player.black else 'white'
    radius = self.padding / 2
    cx = self.window_w / 2
    cy = self.window_h
    self.canvas.create_oval(
      cx - radius, cy - radius, cx + radius, cy + radius,
      fill=color, outline=self.back_ground_color, width=10
    )

  def draw_lines(self):
    for x in range(self.row_n):
      # horizontal lines
      self.canvas.create_line(
        self.padding,
        self.padding + x * self.cell_size,
        self.padding + self.window_w - 2 * self.padding,
        self.padding + x * self.cell_size,
        fill="black"
      )
    for x in range(self.col_n):
      # vertical lines
      self.canvas.create_line(
        self.padding + x * self.cell_size,
        self.padding,
        self.padding + x * self.cell_size,
        self.padding + self.window_h - 2 * self.padding,
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
    if self.row_n == self.col_n == 19:
      points = [3, 9, 15]
    elif self.row_n == self.col_n == 13:
      points = [3, 6, 9]
    elif self.row_n == self.col_n == 9:
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
    for x in range(self.col_n):
      for y in range(self.row_n):
        stone = self.cur_game_state.stones[y][x]
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
    
    self.draw_mcts_q_bar()
    
    best_move_pos = self.cur_mcts_data.best_pos()
    
    for x in range(self.col_n):
      for y in range(self.row_n):
        q, visited_time = self.cur_mcts_data.get(row=y, col=x)
        if visited_time < 5:
          continue
        
        if (y, x) == best_move_pos:
          candidate_color = '#0CE6E6' # HSV: 180 95 90
          self.draw_piece(x, y, '#5A5A5A', 2)
          self.draw_piece(x, y, '#262626', 1)
        else:
          color_table = [
                               #   H  S  V
            (0.1,  '#C02F2A'), #   2 78 75
            (0.3,  '#DA4939'), #   6 74 85
            (0.45, '#DD8829'), #  32 81 87
            (0.5,  '#DACA21'), #  55 85 85
            (0.55, '#ABDA21'), #  76 85 85
            (0.7,  '#66DA2C'), # 100 80 85
            (0.9,  '#249C24'), # 120 77 61
            (1,    '#1D8026')  # 126 77 50
          ]
          candidate_color = color_table[0][1]
          for upper_bound, color in color_table:
            if q <= upper_bound:
              candidate_color = color
              break
        self.draw_piece(x, y, candidate_color)

        cx = self.padding + x * self.cell_size
        cy = self.padding + y * self.cell_size
        vertical_bias = 5
        self.canvas.create_text(cx, cy - vertical_bias, text=f'{q:.2f}', font=(self.font, 10), fill='black')
        self.canvas.create_text(cx, cy + vertical_bias, text=f'{int(visited_time)}', font=(self.font, 10), fill='black')

  def draw_mcts_q_bar(self):
    if self.cur_mcts_data.win_rate is None:
      return
    
    black_win_rate = self.cur_mcts_data.win_rate \
      if self.cur_game_state.next_player == Player.black \
      else 1 - self.cur_mcts_data.win_rate
    white_win_rate = 1 - black_win_rate
    
    width = self.window_w - 2 * self.padding

    x0 = self.padding
    y0 = int(self.window_h - self.winning_rate_bar_height / 2)

    def linear_compress(value, min, max, new_min, new_max):
      assert max > min and new_max > new_min
      return ((value - min) / (max - min)) * (new_max - new_min) + new_min
    
    black_width = width * black_win_rate
    # avoid confusion with winning rate text
    black_width = int(linear_compress(
      black_width, 
      0, width, 
      self.winning_rate_bar_height * 1.6, width - self.winning_rate_bar_height * 1.6
    ))

    self.canvas.create_rectangle(x0, y0, x0 + width, y0 + self.winning_rate_bar_height, fill='white', outline='')
    self.canvas.create_rectangle(x0, y0, x0 + black_width, y0 + self.winning_rate_bar_height, fill='black', outline='')

    gap = self.winning_rate_bar_height // 4

    def render_win_rate_text(player: Player, win_rate: float):
      is_black = player == Player.black
      self.canvas.create_text(
        x0 + gap if is_black else x0 + width - gap, 
        y0 + self.winning_rate_bar_height // 2 - 1, 
        text=f'{win_rate * 100:.1f}', font=(self.font, self.winning_rate_bar_height // 2),
        fill='white' if is_black else 'black', 
        anchor='w' if is_black else 'e'
      )
    
    render_win_rate_text(Player.black, black_win_rate)
    render_win_rate_text(Player.white, white_win_rate)
    
  def draw_message(self):
    if self.cur_game_state.is_over:
      winner = 'white' if self.cur_game_state.winner == Player.white else 'black'
      game_result = self.cur_game_state.game_result
      message = f'{winner} wins  {game_result}'
      self.canvas.create_text(
        self.window_w / 2, self.padding / 2,
        text=message, font=("Arial", 20), fill="red"
      )

  def on_mouse_move(self, event: tk.Event):
    x = round((event.x - self.padding) / self.cell_size)
    y = round((event.y - self.padding) / self.cell_size)
    if 0 <= x < self.col_n and 0 <= y < self.row_n:
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
    if 0 <= x < self.col_n and 0 <= y < self.row_n:
      self.push_play(y + 1, x + 1)

  def on_pass_turn_button(self):
    self.push_pass_turn()

  def on_resign_button(self):
    self.push_resign()

  def on_undo_botton(self):
    self.push_undo()
    
  def on_tab(self, event: tk.Event):
    self.push_pass_turn()
    
  def on_ctrl_z(self, event: tk.Event):
    self.push_undo()
    
  def push_play(self, row, col):
    '''indexing starts from 1'''
    assert self.move_queue is not None
    move = Move.play(Point(row=row, col=col))
    UI.enqueue_move(self.move_queue, move)

  def push_pass_turn(self):
    assert self.move_queue is not None
    move = Move.pass_turn()
    UI.enqueue_move(self.move_queue, move)
    
  def push_resign(self):
    assert self.move_queue is not None
    move = Move.resign()
    UI.enqueue_move(self.move_queue, move)
    
  def push_undo(self):
    assert self.move_queue is not None
    move = Move.undo()
    UI.enqueue_move(self.move_queue, move)
