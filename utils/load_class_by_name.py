import importlib

__all__ = [
  'load_class_by_name'
]

def load_class_by_name(full_class_name: str) -> type:
  module_name, class_name = full_class_name.rsplit('.', 1)
  module = importlib.import_module(module_name)
  cls = getattr(module, class_name)
  return cls