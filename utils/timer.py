from time import time
from typing import Callable

import cProfile
import pstats
import io

__all__ = [
  'timer',
  'profile',
]

def timer(time_precision: int = 2) -> Callable:
  def decorator(func: Callable) -> Callable:
    def decorated_func(*args, **kwargs):
      start_time = time()

      result = func(*args, **kwargs)

      end_time = time()

      ms_cost = (end_time - start_time) * 1000
      ms_cost = round(ms_cost, time_precision)
      print(f'`{func.__name__}` takes {ms_cost}ms')

      return result

    return decorated_func

  return decorator

def profile(func: Callable) -> Callable:
  def decorated_func(*args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    print(stream.getvalue())
  return decorated_func