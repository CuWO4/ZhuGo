import argparse
import json

def main():
  parser = argparse.ArgumentParser(description = 'you can also use command line parameters '
                                   'instead of configuration to set the train; `--optimizer.lr 1e-2` etc.')
  parser.add_argument('-m', '--model', type = str, required = True, help = 'root to model')
  parser.add_argument('-c', '--conf', type = str, required = True, help = 'path to configuration')
  parser.add_argument('-d', '--dataset', type = str, required = True, help = 'root to dataset')
  parser.add_argument('-t', '--test-dataset', type = str, help = 'root to test dataset. '
                      'if not given, test phase is skipped')
  args, unknown_args = parser.parse_known_args()

  with open(args.conf, 'r') as config_file:
    config = json.load(config_file)

  for idx in range(0, len(unknown_args), 2):
    key = unknown_args[idx].lstrip("-")
    if idx + 1 >= len(unknown_args):
      print(f'argument error: `{key}` need value')
      exit(-1)
    value = unknown_args[idx + 1]
    try:
      value = int(value)
    except ValueError:
      value = float(value)

    keys = key.split('.')
    current_conf_layer = config
    for subkey in keys[:-1]:
      current_conf_layer = current_conf_layer[subkey]
      if not isinstance(current_conf_layer, dict):
        print(f'argument error: `{subkey}` is an attribute (in `{key}`)')
        exit(-1)
    current_conf_layer[keys[-1]] = value

  # importing includes torch, slow.
  # import after command line arguments are correct
  from train.trainer import Trainer
  from ai.zhugo import ZhuGo
  from ai.encoder.zhugo_encoder import ZhuGoEncoder
  from ai.manager import ModelManager
  from train.optimizer_manager import OptimizerManager
  from train.dataloader import BGTFDataLoader

  import torch.optim as optim

  trainer = Trainer(
    model_manager = ModelManager(args.model, ZhuGo),
    optimizer_manager = OptimizerManager(args.model, optim.SGD, optim_kwargs = config['optimizer']),
    dataloader = BGTFDataLoader(
      args.dataset,
      config['batch_size'],
      ZhuGoEncoder(device = 'cpu'), # prefetched tensors saved on cpu
      debug = True,
      prefetch_batch = config['prefetch_batch'],
    ),
    test_dataloader = BGTFDataLoader(
      args.test_dataset,
      config['batch_size'],
      ZhuGoEncoder(device = 'cpu'), # prefetched tensors saved on cpu
      prefetch_batch = config['test_prefetch_batch'],
    )
      if args.test_dataset is not None
      else None,
    **config['trainer'],
  )

  trainer.train()

if __name__ == '__main__':
  main()
