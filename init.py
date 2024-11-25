def init():
  # since the chess game will continue to save the previous state, 
  # the maximum recursive depth should be increased
  import sys
  sys.setrecursionlimit(100000)