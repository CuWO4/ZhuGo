import init
import game.ui_game as ui_game
import game.ui_analysis as ui_analysis
from agent.base import Agent
from utils.load_class_by_name import load_class_by_name

import json
import argparse

def parse_args() -> tuple[str]:
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--conf', type=str, 
                      default='conf/main/default.json', 
                      help='path to configuration file, human vs strongest robot so far'
                           'by default. check conf/main/ for more details.')
  args = parser.parse_args()
  
  return (args.conf,)

def create_agents(agents_confs: list[dict]) -> list[Agent]:
  agents = []
  for agent_conf in agents_confs:
    agent = load_class_by_name(agent_conf['class_name'])(**agent_conf['args'])
    agents.append(agent)
  return agents

def get_main_json_conf(conf_path: str) -> tuple[str, list[Agent], type, dict]:
  '''return ((agents), UI Class, game setting dictionary)'''
  with open(conf_path,'r') as config_file:
    config = json.load(config_file)

    mode = config.get('mode')
    agents = create_agents(config.get('agents'))
    UIClass = load_class_by_name(config.get('gui'))
    game_settings = config.get('game')
    
    return mode, agents, UIClass, game_settings
    
def main():
  init.init()
  (conf_path,) = parse_args()
  mode, agents, UIClass, game_settings = get_main_json_conf(conf_path)
  if mode == 'game':
    ui_game.start_game(*agents, UIClass, game_settings)
  elif mode == 'analysis':
    ui_analysis.start_analysis(*agents, UIClass, game_settings)

if __name__ == '__main__':
  main()