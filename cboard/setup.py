from setuptools import setup, Extension
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
filepath = script_dir + '\\boardmodule.c'
# Define the cboard extension
cboard_extension = Extension(
  'boardmodule',
  sources=[filepath],
  extra_compile_args=['-O2'],
  include_dirs=[],
)

# Setup script
setup(
  name='boardmodule',
  version='1.0',
  description='C Board module for Go game',
  ext_modules=[cboard_extension],
)
