from setuptools import setup, Extension
import os

def find_c_files(directory: str) -> list[str]:
  c_files = []
  for root, _, files in os.walk(directory):
    for file in files:
      if file.endswith('.c') or file.endswith('.cc') or file.endswith('.cpp'):
        c_files.append(os.path.join(root, file))
  return c_files

script_dir = os.path.dirname(os.path.abspath(__file__))
c_files = find_c_files(script_dir)
c_files += find_c_files(script_dir + "/../cboard")

import numpy
numpy_dir = os.path.dirname(numpy.__file__)

cboard_extension = Extension(
  'encodermodule',
  sources=c_files,
  extra_compile_args=['-O2', '-DNDEBUG', '/std:c++17'],
  include_dirs=[
    script_dir + '/../cboard/',
    numpy_dir + '/core/include'
  ],
)

setup(
  name='encoder',
  version='1.0',
  description='C Encoder module for ZhuGo',
  ext_modules=[cboard_extension],
)
