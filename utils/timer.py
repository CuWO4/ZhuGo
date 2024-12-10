from time import time
from typing import Callable

__all__ = [
  'timer'
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
  