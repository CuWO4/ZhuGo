import subprocess
import zstandard as zstd
from argparse import ArgumentParser
import os

DEBUGDIR = 'debug'
TARGET = 'leela_to_bgtf.exe'


if __name__ == '__main__':
  current_dir = os.getcwd()
  script_dir = os.path.dirname(os.path.abspath(__file__))

  subprocess.run(['make', '-C', script_dir, f'DEBUGDIR={DEBUGDIR}', f'TARGET={TARGET}'])

  parser = ArgumentParser(description='compress other train set format to ZhuGo train set format BGTF')

  parser.add_argument('work_directory')
  parser.add_argument('output_directory')
  parser.add_argument('--type', choices=['leela'], help='source format type')
  parser.add_argument('--level', type=int, default=3, help='compress level (3 by default)')

  args = parser.parse_args()

  cctx = zstd.ZstdCompressor(level = args.level)

  executable = os.path.join(script_dir, os.path.join(DEBUGDIR, TARGET))
  for dir, _, files in os.walk(args.work_directory):
    for file in files:
      leela_path = os.path.join(current_dir, dir, file)
      bgtf_path = os.path.join(current_dir, args.output_directory, file.split('.')[0] + '.bgtf')
      bgtf_zstd_path = os.path.join(current_dir, args.output_directory, file.rsplit('.')[0] + '.bgtf.zstd')
      subprocess.run([executable, leela_path, bgtf_path])
      with open(bgtf_path, 'rb') as f_in, open(bgtf_zstd_path, 'wb') as f_out:
        f_out.write(cctx.compress(f_in.read()))
      os.remove(bgtf_path)
      print(f'compressed {leela_path} -> {bgtf_zstd_path}')
