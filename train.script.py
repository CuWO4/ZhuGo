import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', type = str, required = True, help = 'root to model')
  parser.add_argument('--batch-size', type = int, default = 128)
  parser.add_argument('--batch-accumulation', type = int, default = 1,
                      help = 'equivalent virtual batch size = batch size * batch accumulation')
  # supervisor learning does not requires very large batch size
  parser.add_argument('-d', '--dataset', type = str, required = True, help = 'root to dataset')
  parser.add_argument('--batch-per-test', type = int, default = 100)
  parser.add_argument('--test-dataset', type = str, help = 'root to test dataset. if not given, '
                      'test phase is skipped')
  parser.add_argument('--prefetch-batch', type = int, default = 300)
  parser.add_argument('--test-prefetch-batch', type = int, default = 20)
  parser.add_argument('--lr', type = float, default = 0.1, help = 'base lr')
  parser.add_argument('--weight-decay', type = float, default = 1e-4)
  parser.add_argument('--momentum', type = float, default = 0.9)
  parser.add_argument('--gradient-clip', type = float, default = 2.0)
  parser.add_argument('--policy-loss-weight', type = float, default = 0.85)
  parser.add_argument('--value-loss-weight', type = float, default = 0.03)
  parser.add_argument('--soft-target-nominal-weight', type = float, default = 2.0,
                      help = 'nominal weight of soft policy target')
  parser.add_argument('--softening-intensity', type = float, default = 1 / 4,
                      help = 'softening temperature / original temperature')
  parser.add_argument('--checkpoint-interval-sec', type = int, default = 3600, help = 'seconds '
                      'between two checkpoints saved')
  args = parser.parse_args()

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
    optimizer_manager = OptimizerManager(
      args.model, optim.SGD,
      optim_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
      }
    ),
    dataloader = BGTFDataLoader(
      args.dataset,
      args.batch_size,
      ZhuGoEncoder(device = 'cpu'), # prefetched tensors saved on cpu
      debug = True,
      prefetch_batch = args.prefetch_batch,
    ),
    batch_accumulation = args.batch_accumulation,
    batch_per_test = args.batch_per_test,
    test_dataloader = BGTFDataLoader(
      args.test_dataset,
      args.batch_size,
      ZhuGoEncoder(device = 'cpu'), # prefetched tensors saved on cpu
      prefetch_batch = args.test_prefetch_batch,
    )
      if args.test_dataset is not None
      else None,
    gradient_clip = args.gradient_clip,
    policy_loss_weight = args.policy_loss_weight,
    value_loss_weight = args.value_loss_weight,
    soft_target_nominal_weight = args.soft_target_nominal_weight,
    softening_intensity = args.softening_intensity,
    checkpoint_interval_sec = args.checkpoint_interval_sec,
  )

  trainer.train()

if __name__ == '__main__':
  main()