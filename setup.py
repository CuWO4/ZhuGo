import os
import subprocess
from contextlib import contextmanager
import sys

@contextmanager
def switch_workdir(target_dir: str, debug: bool = False):
  current_dir = os.getcwd()
  target_dir = os.path.abspath(target_dir)

  try:
    os.chdir(target_dir)
    if debug: print(f'<{__file__}> enter {target_dir}')
    yield
  except FileNotFoundError:
    print(f"dir not exists: `{target_dir}`")
  finally:
    os.chdir(current_dir)
    if debug: print(f'<{__file__}> leave {target_dir}')

def main():
  with switch_workdir('cboard', debug = True):
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'])
  with switch_workdir('cencoder', debug = True):
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'])

if __name__ == '__main__':
  main()
