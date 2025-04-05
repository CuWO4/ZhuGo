from train.meta import MetaData
from conf.utils import GeneralObjectEncoder, general_object_hook

import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import contextlib

__all__ = [
  'create',
  'save',
  'load',
]


MODEL_FILE = 'model.model'
MODEL_SETTING_FILE = 'model.json'
TENSORBOARD_DIR = 'tensorboard'
META_DATA_FILE = 'meta.json'
DEPLOY_FILE = 'model.pt'

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

  meta = MetaData()
  meta_json = os.path.join(path, META_DATA_FILE)
  with open(meta_json, "w") as f:
    json.dump(meta, f, cls=GeneralObjectEncoder)

  print('model successfully created')

def save(model: torch.nn.Module, path: str):
  model_file = os.path.join(path, MODEL_FILE)
  torch.save(model.state_dict(), model_file)

  print('model successfully saved')

def load(
  ModelType: type, path: str,
  device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.nn.Module:
  model_json = os.path.join(path, MODEL_SETTING_FILE)
  with open(model_json, "r") as f:
    model_params = json.load(f)

  model = ModelType(**model_params)
  model_file = os.path.join(path, MODEL_FILE)
  model.load_state_dict(torch.load(model_file))
  model = model.to(device=device)

  print('model successfully loaded')

  return model

def save_meta(meta: MetaData, path: str):
  meta_json = os.path.join(path, META_DATA_FILE)
  with open(meta_json, "w") as f:
    json.dump(meta, f, cls=GeneralObjectEncoder)

  print('meta of model successfully saved')

def load_meta(path: str) -> MetaData:
  meta_json = os.path.join(path, META_DATA_FILE)
  with open(meta_json, "r") as f:
    meta = json.load(f, object_hook=general_object_hook)

  print('meta of model successfully loaded')

  return meta

@contextlib.contextmanager
def load_summary_writer(path: str) -> SummaryWriter:
  tb_dir = os.path.join(path, TENSORBOARD_DIR)
  writer = SummaryWriter(tb_dir)
  print('summary writer successfully loaded')
  
  try:
    yield writer
  finally:
    writer.close()
    print('summary writer closed')

def deploy(ModelType: type, path: str, dumb_input: torch.Tensor):
  '''generate static TorchScript version of model'''
  model = load(ModelType, path, device = 'cpu')

  model.eval()
  with torch.no_grad():
    dumb_input = dumb_input.cpu()
    jit_model = torch.jit.trace(model, dumb_input)

  jit_model_path = os.path.join(path, DEPLOY_FILE)
  jit_model.save(jit_model_path)

  print('model deployed')

def load_deployed_model(
  path: str, 
  device = 'cuda' if torch.cuda.is_available() else 'cpu',
  warmup_dumb_input: torch.Tensor | None = None
) -> torch.jit.ScriptModule:
  jit_model_path = os.path.join(path, DEPLOY_FILE)
  model = torch.jit.load(jit_model_path).to(device = device)

  if warmup_dumb_input is not None:
    warmup_dumb_input = warmup_dumb_input.to(device = device)
    model(warmup_dumb_input)

  print('deployed model successfully loaded')

  return model
