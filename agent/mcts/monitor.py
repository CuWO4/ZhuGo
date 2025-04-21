import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from multiprocessing import connection
from matplotlib.ticker import MaxNLocator

class Monitor:
  def __init__(self, data_connection: connection.Connection):
    self.data_connection = data_connection

    self.background_color = '#CDAC6A'
    self.font = ('Consolas', 12)

    self.root = tk.Tk()
    self.root.title("MCTS Monitor")

    self.text_frame = ttk.Frame(self.root)
    self.text_frame.grid(row=0, column=0, padx=5, sticky="w")

    self.entropy_label = ttk.Label(self.text_frame, text="Entropy: ", font = self.font)
    self.entropy_label.grid(row=0, column=0, sticky='w')
    self.visited_time_sum_label = ttk.Label(self.text_frame, text="Visited Time Sum: ", font = self.font)
    self.visited_time_sum_label.grid(row=1, column=0, sticky='w')
    self.time_cost_label = ttk.Label(self.text_frame, text="Time Cost (s): ", font = self.font)
    self.time_cost_label.grid(row=2, column=0, sticky='w')

    self.plot_frame = ttk.Frame(self.root)
    self.plot_frame.grid(row=1, column=0)

    self.turns = []
    self.black_win_rates = []
    self.white_win_rates = []

    self.fig, self.ax = plt.subplots(figsize=(4, 3))
    self.line_black, = self.ax.plot(self.turns, self.black_win_rates, color='black')
    self.line_white, = self.ax.plot(self.turns, self.white_win_rates, color='white')

    self.ax.set_ylim(0, 1)
    self.ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[1]))

    self.ax.grid(True, color = '#B3975D')

    self.fig.patch.set_facecolor(self.background_color)
    self.ax.set_facecolor(self.background_color)
    self.fig.subplots_adjust(bottom=0.15)

    self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack()

    self.root.after(50, self.update_plot)

    self.root.mainloop()

  def update_plot(self):
    self.update_data()

    self.line_black.set_xdata(self.turns)
    self.line_black.set_ydata(self.black_win_rates)

    self.line_white.set_xdata(self.turns)
    self.line_white.set_ydata(self.white_win_rates)

    self.ax.relim()
    self.ax.autoscale_view()

    self.canvas.draw()

    self.root.after(50, self.update_plot)

  def update_data(self):
    updated = self.data_connection.poll()
    while self.data_connection.poll():
      turn, black_win_rate, entropy, visited_time_sum, time_cost_sec = self.data_connection.recv()

    if updated:
      self.update_text(entropy, visited_time_sum, time_cost_sec)
      self.update_black_win_rates(turn, black_win_rate)

  def update_text(self, entropy, visited_time_sum, time_cost_sec):
    self.entropy_label.config(text=f"Entropy: {entropy:.2f}")
    self.visited_time_sum_label.config(text=f"Visited Time Sum: {visited_time_sum}")
    self.time_cost_label.config(text=f"Time Cost: {time_cost_sec}s")

  def update_black_win_rates(self, turn, black_win_rate):
    while turn > len(self.turns):
      self.turns.append(turn)
      self.black_win_rates.append(None)
      self.white_win_rates.append(None)

    self.black_win_rates[turn - 1] = black_win_rate
    self.white_win_rates[turn - 1] = 1 - black_win_rate

