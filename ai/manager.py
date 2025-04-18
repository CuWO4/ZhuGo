from train.meta import MetaData
from conf.utils import GeneralObjectEncoder, general_object_hook

import os
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from contextlib import contextmanager
from datetime import datetime

__all__ = [
  'ModelManager',
]

class ModelManager:
  MODEL_FILE = 'model.model'
  MODEL_SETTING_FILE = 'model.json'
  META_DATA_FILE = 'meta.json'
  CHECKPOINTS_DIR = 'checkpoints'
  TENSORBOARD_DIR = 'tensorboard'
  DEPLOY_FILE = 'model.pt'

  def __init__(self, root: str, ModelType: type):
    self.root = root
    self.ModelType = ModelType

  def create(self, model_settings: dict, dumb_input: torch.Tensor):
    try:
      os.makedirs(self.root)
    except OSError:
      while True:
        choice = input(f'`{self.root}` already exists, overwrite? [y/N]: ')
        if choice == 'y' or choice == 'Y': break
        if choice == 'n' or choice == 'N' or choice == '': return

    model = self.ModelType(**model_settings)
    model_path = os.path.join(self.root, self.MODEL_FILE)
    torch.save(model.state_dict(), model_path)

    model_setting_path = os.path.join(self.root, self.MODEL_SETTING_FILE)
    with open(model_setting_path, "w") as f:
      json.dump(model_settings, f)

    meta = MetaData()
    meta_json = os.path.join(self.root, self.META_DATA_FILE)
    with open(meta_json, 'w') as f:
      json.dump(meta, f, cls=GeneralObjectEncoder)

    checkpoints_path = os.path.join(self.root, self.CHECKPOINTS_DIR)
    os.makedirs(checkpoints_path, exist_ok=True)

    tensorboard_path = os.path.join(self.root, self.TENSORBOARD_DIR)
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(tensorboard_path)
    model = model.cpu()
    dumb_input = dumb_input.cpu()
    writer.add_graph(model, dumb_input)
    writer.close()

    print('model successfully created')

  def save_model(self, model: nn.Module):
    model_path = os.path.join(self.root, self.MODEL_FILE)
    torch.save(model.state_dict(), model_path)

    print('model successfully saved')

  def load_model(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> nn.Module:
    model_setting_path = os.path.join(self.root, self.MODEL_SETTING_FILE)
    with open(model_setting_path, "r") as f:
      model_settings = json.load(f)

    model = self.ModelType(**model_settings)
    model_path = os.path.join(self.root, self.MODEL_FILE)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device=device)

    print('model successfully loaded')

    return model

  def save_meta(self, meta: MetaData):
    meta_path = os.path.join(self.root, self.META_DATA_FILE)
    with open(meta_path, "w") as f:
      json.dump(meta, f, cls=GeneralObjectEncoder)

    print('meta of model successfully saved')

  def load_meta(self) -> MetaData:
    meta_json = os.path.join(self.root, self.META_DATA_FILE)
    with open(meta_json, "r") as f:
      meta = json.load(f, object_hook=general_object_hook)

    print('meta of model successfully loaded')

    return meta
  
  def save_checkpoint(self, model: nn.Module):
    checkpoint_path = os.path.join(
      self.root,
      self.CHECKPOINTS_DIR,
      datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    )
    torch.save(model.state_dict(), checkpoint_path)

    print('checkpoint successfully saved')

  @contextmanager
  def load_summary_writer(self) -> SummaryWriter:
    tensorboard_dir = os.path.join(self.root, self.TENSORBOARD_DIR)
    writer = SummaryWriter(tensorboard_dir)
    print('summary writer successfully loaded')

    try:
      yield writer
    finally:
      writer.close()
      print('summary writer closed')

  def deploy(self, dumb_input: torch.Tensor):
    '''generate static TorchScript version of model'''
    model = self.load_model(device = 'cpu')

    model.eval()
    with torch.no_grad():
      dumb_input = dumb_input.cpu()
      jit_model = torch.jit.trace(model, dumb_input)

    jit_model_path = os.path.join(self.root, self.DEPLOY_FILE)
    jit_model.save(jit_model_path)

    print('model deployed')

  def load_deployed_model(
    self,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    warmup_dumb_input: torch.Tensor | None = None
  ) -> torch.jit.ScriptModule:
    jit_model_path = os.path.join(self.root, self.DEPLOY_FILE)
    model = torch.jit.load(jit_model_path).to(device = device)

    if warmup_dumb_input is not None:
      model(warmup_dumb_input.to(device = device))

    print('deployed model successfully loaded')

    return model
