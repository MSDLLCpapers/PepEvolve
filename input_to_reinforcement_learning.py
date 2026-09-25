import json
import time 
import logging
import os
from sys import argv

from manager import Manager
from pepinvent.reinforcement.configuration.reinforcement_learning_configuration import \
                                                                                       ReinforcementLearningConfiguration


def read_json_file(path):
    with open(path) as f:
        json_input = f.read().replace('\r', '').replace('\n', '')
    try:
        return json.loads(json_input)
    except (ValueError, KeyError, TypeError) as e:
        print(f"JSON format error in file ${path}: \n ${e}")

def customize(learning_parameters, name) : 
    # Turn off lightning log
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    learning_parameters.logging.logging_path = current_dir + learning_parameters.logging.logging_path + name
    learning_parameters.logging.result_path = current_dir + learning_parameters.logging.result_path + name 

    return learning_parameters 
    

if __name__ == "__main__":

    path, name = argv[1], argv[2]
    config = read_json_file(path)

    learning_parameters = ReinforcementLearningConfiguration.parse_obj(config)
    learning_parameters = customize(learning_parameters, name)

    manager = Manager(learning_parameters)

    start_time = time.time()
    manager.execute()
    end_time = time.time()

    with open(f'{learning_parameters.logging.logging_path}/execution_time.txt', 'w') as f:
        f.write(str(end_time - start_time))