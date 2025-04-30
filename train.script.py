import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', type = str, required = True, help = 'root to model')
  parser.add_argument('--batch-size', type = int, default = 128)
  # supervisor learning does not requires very large batch size
  parser.add_argument('-d', '--dataset', type = str, required = True, help = 'root to dataset')
  parser.add_argument('--batch-per-test', type = int, default = 100)
  parser.add_argument('--test-dataset', type = str, help = 'root to test dataset. if not given, '
                      'test phase is skipped')
  parser.add_argument('--lr', type = float, default = 0.1, help = 'base lr')
  parser.add_argument('--weight-decay', type = float, default = 1e-4)
  parser.add_argument('--momentum', type = float, default = 0.9)
  parser.add_argument('--gradient-clip', type = float, default = 2.0)
  parser.add_argument('--T_max', type = int, default = 10000, help = 'T_max of cosine lr schedular')
  parser.add_argument('--eta_min', type = float, default = 1e-4, help = 'least lr')
  parser.add_argument('--policy-loss-weight', type = float, default = 0.85)
  parser.add_argument('--value-loss-weight', type = float, default = 0.03)
  parser.add_argument('--checkpoint-interval-sec', type = int, default = 3600, help = 'seconds '
                      'between two checkpoints saved')
  args = parser.parse_args()

  # importing includes torch, slow.
  # import after command line arguments are correct
  from train.trainer import Trainer
  from ai.zhugo import ZhuGo
  from ai.encoder.zhugo_encoder import ZhuGoEncoder
  from ai.manager import ModelManager
  from train.dataloader import BGTFDataLoader


  trainer = Trainer(
    model_manager = ModelManager(args.model, ZhuGo),
    dataloader = BGTFDataLoader(
      args.dataset,
      args.batch_size,
      ZhuGoEncoder(device = 'cpu'), # prefetched tensors saved on cpu
      debug = True,
    ),
    batch_per_test = args.batch_per_test,
    test_dataloader = BGTFDataLoader(
      args.test_dataset,
      64,
      ZhuGoEncoder(device = 'cpu'), # prefetched tensors saved on cpu
    )
      if args.test_dataset is not None
      else None,
    base_lr = args.lr,
    weight_decay = args.weight_decay,
    momentum = args.momentum,
    gradient_clip = args.gradient_clip,
    T_max = args.T_max,
    eta_min = args.eta_min,
    policy_loss_weight = args.policy_loss_weight,
    value_loss_weight = args.value_loss_weight,
    checkpoint_interval_sec = args.checkpoint_interval_sec,
  )

  trainer.train()

if __name__ == '__main__':
  main()