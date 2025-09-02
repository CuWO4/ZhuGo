import init

import json
import argparse

def start_play(args: argparse.Namespace, conf: dict):
  import game.ui_game as ui_game
  import game.ui_analysis as ui_analysis
  from utils.load_class_by_name import load_class_by_name

  agents = [
    load_class_by_name(agent_conf['class_name'])(**agent_conf['args'])
    for agent_conf in conf['agents']
  ]

  if conf['mode'] == 'game':
    ui_game.start_game(
      *agents,
      UIClass = load_class_by_name(conf['gui']),
      game_settings = conf['game']
    )
  elif conf['mode'] == 'analysis':
    ui_analysis.start_analysis(
      *agents,
      UIClass = load_class_by_name(conf['gui']),
      game_settings = conf['game']
    )
  else:
    print(f'invalid play mode {conf["mode"]}')
    exit(-1)

def start_create(args: argparse.Namespace, conf: dict):
  from go.goboard import GameState
  from ai.zhugo import ZhuGo
  from ai.manager import ModelManager
  from ai.encoder.zhugo_encoder import ZhuGoEncoder

  dumb_input = ZhuGoEncoder().encode(GameState.new_game(conf['board_size'])).unsqueeze(0)

  manager = ModelManager(args.path, ZhuGo)
  manager.create(model_settings = conf, dumb_input = dumb_input)

def start_train(args: argparse.Namespace, conf: dict):
  from train.trainer import Trainer
  from ai.zhugo import ZhuGo
  from ai.encoder.zhugo_encoder import ZhuGoEncoder
  from ai.manager import ModelManager
  from train.optimizer_manager import OptimizerManager
  from train.dataloader import BGTFDataLoader

  import torch.optim as optim

  trainer = Trainer(
    model_manager = ModelManager(args.model, ZhuGo),
    optimizer_manager = OptimizerManager(
      args.model, optim.SGD, **conf['optimizer']['lr'],
      optim_kwargs = conf['optimizer']['arguments']
    ),
    dataloader = BGTFDataLoader(
      args.dataset,
      conf['batch_size'],
      ZhuGoEncoder(device = 'cpu'), # prefetched tensors saved on cpu
      debug = True,
      prefetch_batch = conf['prefetch_batch'],
    ),
    test_dataloader = BGTFDataLoader(
      args.test_dataset,
      conf['batch_size'],
      ZhuGoEncoder(device = 'cpu'), # prefetched tensors saved on cpu
      prefetch_batch = conf['test_prefetch_batch'],
    )
      if args.test_dataset is not None
      else None,
    **conf['trainer'],
  )

  trainer.train()

def parse_args() -> tuple[argparse.Namespace, dict]:
  '''return args, conf'''
  parser = argparse.ArgumentParser(
    description = 'you can also use command line parameters instead of '
    'confuration to set the train; `--optimizer.lr 1e-2` or `--agents.0.args.model ./path/to/model` etc.'
  )
  subparsers = parser.add_subparsers(dest = 'command', required = True)

  play_subparser = subparsers.add_parser('play')
  play_subparser.add_argument(
    '-c', '--conf', type = str, required = True,
    help='path to confuration file, check conf/main/ for more details.'
  )

  create_subparser = subparsers.add_parser('create')
  create_subparser.add_argument(
    '-p', '--path', type = str, required = True,
    help='path to model.'
  )
  create_subparser.add_argument(
    '-c', '--conf', type = str, required = True,
    help='path to confuration file, check conf/model/ for more details.'
  )

  train_subparser = subparsers.add_parser('train')
  train_subparser.add_argument(
    '-m', '--model', type = str, required = True,
    help = 'root to model'
  )
  train_subparser.add_argument(
    '-c', '--conf', type = str, required = True,
    help = 'path to confuration file, check conf/train/ for more details'
  )
  train_subparser.add_argument(
    '-d', '--dataset', type = str, required = True,
    help = 'root to dataset'
  )
  train_subparser.add_argument(
    '-t', '--test-dataset', type = str,
    help = 'root to test dataset. if not given, test phase is skipped'
  )

  args, unknown_args = parser.parse_known_args()

  with open(args.conf, 'r') as conf_file:
    conf = json.load(conf_file)

  # cover the conf arguments with cmd line arguments with same name
  for idx in range(0, len(unknown_args), 2):
    key = unknown_args[idx].lstrip("-")
    if idx + 1 >= len(unknown_args):
      print(f'argument error: `{key}` need value')
      exit(-1)
    value = unknown_args[idx + 1]

    try: value = float(value)
    except ValueError: ...

    try: value = int(value)
    except ValueError: ...

    keys = key.split('.')
    current_conf_layer = conf
    for idx, subkey in enumerate(keys[:-1]):
      parent_key = keys[idx - 1] if idx - 1 >= 0 else 'root'
      if isinstance(current_conf_layer, dict):
        current_conf_layer = current_conf_layer[subkey]
      elif isinstance(current_conf_layer, list):
        try:
          current_conf_layer = current_conf_layer[int(subkey)]
        except ValueError:
          print(f'`{parent_key}` is a list, expected a int index but get `{subkey}` (int `{key}`)')
          exit(-1)
      else:
        print(f'argument error: `{parent_key}` is an attribute (in `{key}`)')
        exit(-1)
    current_conf_layer[keys[-1]] = value

  return args, conf

def main():
  init.init()
  args, conf = parse_args()

  if args.command == 'play':
    start_play(args, conf)
  elif args.command == 'create':
    start_create(args, conf)
  elif args.command == 'train':
    start_train(args, conf)
  else:
    assert False, 'impossible'

if __name__ == '__main__':
  main()
