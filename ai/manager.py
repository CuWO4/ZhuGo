import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter

__all__ = [
  'create',
  'save',
  'load',
]

MODEL_FILE = 'model.model'
MODEL_SETTING_FILE = 'model.json'
TENSORBOARD_DIR = 'tensorboard'
META_DATA_FILE = 'meta.json'

def create(ModelType: type, model_params: dict, path: str, dumb_input: torch.Tensor) -> None:
  os.makedirs(path)
  
  model = ModelType(**model_params)
  model_file = os.path.join(path, MODEL_FILE)
  torch.save(model.state_dict(), model_file)
  
  model_json = os.path.join(path, MODEL_SETTING_FILE)
  with open(model_json, "w") as f:
    json.dump(model_params, f)
  
  tb_dir = os.path.join(path, TENSORBOARD_DIR)
  os.makedirs(tb_dir, exist_ok=True)
  writer = SummaryWriter(tb_dir)
  
  model = model.cpu()
  dumb_input = dumb_input.cpu()
  writer.add_graph(model, dumb_input)
  writer.close()
  
  meta = { "epoch": 0 }
  meta_json = os.path.join(path, META_DATA_FILE)
  with open(meta_json, "w") as f:
    json.dump(meta, f)
    
  print('model successfully created')

def save(model: torch.nn.Module, meta: dict, path: str):
  model_file = os.path.join(path, MODEL_FILE)
  torch.save(model.state_dict(), model_file)
  
  meta_json = os.path.join(path, META_DATA_FILE)
  with open(meta_json, "w") as f:
    json.dump(meta, f)
    
  print('model successfully saved')

def load(
  ModelType: type, path: str, 
  device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> tuple[torch.nn.Module, dict, SummaryWriter]:
  model_json = os.path.join(path, MODEL_SETTING_FILE)
  print(model_json)
  with open(model_json, "r") as f:
    model_params = json.load(f)
  
  model = ModelType(**model_params)
  model_file = os.path.join(path, MODEL_FILE)
  model.load_state_dict(torch.load(model_file))
  model = model.to(device=device)
  
  meta_json = os.path.join(path, META_DATA_FILE)
  with open(meta_json, "r") as f:
    meta = json.load(f)
  
  tb_dir = os.path.join(path, TENSORBOARD_DIR)
  writer = SummaryWriter(tb_dir)
  
  print('model successfully loaded')
  
  return model, meta, writer