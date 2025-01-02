import json

from utils.load_class_by_name import load_class_by_name

__all__ = [
  'GeneralObjectEncoder',
  'general_object_hook'
]

class GeneralObjectEncoder(json.JSONEncoder):
  def default(self, obj):
    return {
      "__classname__" :  '.'.join([obj.__module__, obj.__class__.__name__]),
      "__data__"      :  obj.__dict__,
    }

def general_object_hook(dct):
  if "__classname__" in dct:
    cls = load_class_by_name(dct['__classname__'])
    obj = cls()
    obj.__dict__.update(dct["__data__"])
    return obj
  return dct
