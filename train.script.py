import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', type = str, required = True, help = 'root to model')
  parser.add_argument('--batch-size', type = int, default = 128)
  # supervisor learning does not requires very large batch size
  parser.add_argument('-d', '--dataset', type = str, required = True, help = 'root to dataset')
  parser.add_argument('--refuel-records', type = int, default = 2000, help = 'count of records '
                      'in each refueling')
  parser.add_argument('-c', '--capacity', type = int, default = 2001, help = 'capacity of '
                      'exp pool') # to avoid periodic zero values ​​due to pool emptying
  parser.add_argument('--batch-per-refuel', type = int, default = 16)
  parser.add_argument('--test-dataset', type = str, help = 'root to test dataset. if not given, '
                      'test phase is skipped')
  parser.add_argument('--lr', type = float, default = 0.1, help = 'base lr')
  parser.add_argument('--weight-decay', type = float, default = 1e-4)
  parser.add_argument('--momentum', type = float, default = 0.9)
  parser.add_argument('--T_max', type = int, default = 10000, help = 'T_max of cosine lr schedular')
  parser.add_argument('--eta_min', type = float, default = 1e-4, help = 'least lr')
  parser.add_argument('--policy-loss-weight', type = float, default = 0.7)
  parser.add_argument('--value-loss-weight', type = float, default = 0.3)
  parser.add_argument('--checkpoint-interval-sec', type = int, default = 3600, help = 'seconds '
                      'between two checkpoints saved')
  args = parser.parse_args()

  # importing includes torch, slow.
  # import after command line arguments are correct
  from train.trainer import Trainer
  from ai.zhugo import ZhuGo
  from ai.encoder.zhugo_encoder import ZhuGoEncoder
  from ai.manager import ModelManager
  from train.exp_pool import ExpPool
  from train.dataloader import BGTFDataLoader


  trainer = Trainer(
    model_manager = ModelManager(args.model, ZhuGo),
    batch_size = args.batch_size,
    dataloader = BGTFDataLoader(
      args.dataset,
      args.refuel_records,
      ZhuGoEncoder(device = 'cpu'), # use to refuel exp pool
      debug = True,
    ),
    exp_pool = ExpPool(args.capacity),
    batch_per_refuel = args.batch_per_refuel,
    test_dataloader = BGTFDataLoader(
      args.test_dataset,
      64,
      ZhuGoEncoder(device = 'cpu'), # use to refuel exp pool
    )
      if args.test_dataset is not None
      else None,
    base_lr = args.lr,
    weight_decay = args.weight_decay,
    momentum = args.momentum,
    T_max = args.T_max,
    eta_min = args.eta_min,
    policy_loss_weight = args.policy_loss_weight,
    value_loss_weight = args.value_loss_weight,
    checkpoint_interval_sec = args.checkpoint_interval_sec,
  )

  trainer.train()

if __name__ == '__main__':
  main()