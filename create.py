from go.goboard import GameState

import json
import argparse

def parse_args() -> tuple[str]:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-p', '--path', type=str, required=True,
    help='path to model.'
  )
  parser.add_argument(
    '-c', '--conf', type=str, required=True,
    help='path to configuration file, check conf/model/ for more details.'
  )
  args = parser.parse_args()
  
  return args.path, args.conf

def main():
  path, conf_path = parse_args()

  with open(conf_path, 'r') as config_file:
    model_settings = json.load(config_file)
  
  # importing includes torch, which is significantly slow
  # import after when command format is correct
  from ai.zhugo import ZhuGo
  from ai.manager import create
  from ai.encoder.zhugo_encoder import ZhuGoEncoder

  dumb_input = ZhuGoEncoder().encode(GameState.new_game(model_settings['board_size'])).unsqueeze(0)
    
  create(ZhuGo, model_settings, path, dumb_input)

if __name__ == '__main__':
  main()
