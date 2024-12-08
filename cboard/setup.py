from setuptools import setup, Extension
import os

def find_c_files(directory: str) -> list[str]:
  c_files = []
  for root, _, files in os.walk(directory):
    for file in files:
      if file.endswith('.c'):
        c_files.append(os.path.join(root, file))
  return c_files

script_dir = os.path.dirname(os.path.abspath(__file__))
c_files = find_c_files(script_dir)
print(c_files)
cboard_extension = Extension(
  'boardmodule',
  sources=c_files,
  extra_compile_args=['-O2', '-DNDEBUG'],
  include_dirs=[],
)

setup(
  name='boardmodule',
  version='1.0',
  description='C Board module for Go game',
  ext_modules=[cboard_extension],
)
