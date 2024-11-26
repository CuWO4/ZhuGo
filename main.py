import init
import game.ui_game as ui_game
from agent.base import Agent

import json
import importlib
import argparse
import agent.mcts_agent_complete
def parse_args() -> tuple[str]:
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--conf', type=str, 
                      default='conf/main/default.json', 
                      help='path to configuration file, human vs strongest robot so far'
                           'by default. check conf/main/ for more details.')
  args = parser.parse_args()
  
  return (args.conf,)

def load_class_by_name(full_class_name: str) -> type:
  module_name, class_name = full_class_name.rsplit('.', 1)
  module = importlib.import_module(module_name)
  cls = getattr(module, class_name)
  return cls

def create_agent(agent_conf: dict) -> Agent:
  agent = load_class_by_name(agent_conf['class_name'])(**agent_conf['args'])
  return agent

def get_main_json_conf(conf_path: str) -> tuple[tuple[Agent], type, dict]:
  '''return ((agents), UI Class, game setting dictionary)'''
  with open(conf_path,'r') as config_file:
    config = json.load(config_file)

    agent1 = create_agent(config.get('agent1'))
    agent2 = create_agent(config.get('agent2'))
    UIClass = load_class_by_name(config.get('gui'))
    game_setting = config.get('game')
    
    return (agent1, agent2), UIClass, game_setting
    
def main():
  init.init()
  (conf_path,) = parse_args()
  (agent1, agent2), UIClass, game_setting = get_main_json_conf(conf_path)
  ui_game.start_game(agent1, agent2, UIClass, **game_setting)

if __name__ == '__main__':
  main()