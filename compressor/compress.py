import subprocess
import zstandard as zstd
from argparse import ArgumentParser
import os
import multiprocessing as mp

DEBUGDIR = 'debug'
TARGET = 'leela_to_bgtf.exe'


def compress_task(
  executable: str,
  work_dir: str,
  output_dir: str,
  files: list[str],
  level: int,
  total: int,
  handled: mp.Value,
  handled_lock: mp.Lock
):
  cctx = zstd.ZstdCompressor(level = level)

  for file in files:
    leela_path = os.path.join(work_dir, file)
    bgtf_path = os.path.join(output_dir, file.split('.')[0] + '.bgtf')
    bgtf_zstd_path = os.path.join(output_dir, file.rsplit('.')[0] + '.bgtf.zstd')
    subprocess.run([executable, leela_path, bgtf_path])
    with open(bgtf_path, 'rb') as f_in, open(bgtf_zstd_path, 'wb') as f_out:
      f_out.write(cctx.compress(f_in.read()))
    os.remove(bgtf_path)

    with handled_lock:
      handled.value += 1
      print(f'{f"compressed {leela_path} -> {bgtf_zstd_path}":<160}{f"{handled.value}/{total}":>30}')

def main():
  current_dir = os.getcwd()
  script_dir = os.path.dirname(os.path.abspath(__file__))

  subprocess.run(['make', '-C', script_dir, f'DEBUGDIR={DEBUGDIR}', f'TARGET={TARGET}'])

  parser = ArgumentParser(description='compress other train set format to ZhuGo train set format BGTF')

  parser.add_argument('work_directory')
  parser.add_argument('output_directory')
  parser.add_argument('--type', choices=['leela'], help='source format type')
  parser.add_argument('--level', type=int, default=3, help='compress level (3 by default)')

  args = parser.parse_args()

  executable = os.path.join(script_dir, os.path.join(DEBUGDIR, TARGET))


  thread_n = mp.cpu_count() + 1
  pool = mp.Pool(thread_n)

  all_files = os.listdir(args.work_directory)

  total = len(all_files)

  work_dir = os.path.join(current_dir, args.work_directory)
  output_dir = os.path.join(current_dir, args.output_directory)

  file_n_each_task = (total + thread_n - 1) // thread_n

  with mp.Manager() as manager:
    handled = manager.Value('i', 0)
    handled_lock = manager.Lock()
    
    pool.starmap(
      compress_task,
      [
        (
          executable,
          work_dir,
          output_dir,
          all_files[idx * file_n_each_task : min((idx + 1) * file_n_each_task, total)],
          args.level,
          total,
          handled,
          handled_lock
        )
        for idx in range(thread_n)
      ]
    )

if __name__ == '__main__':
  main()
