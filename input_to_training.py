import json
from sys import argv

from manager import Manager
from pepinvent.supervised_learning.trainer.architecturedto import ArchitectureConfig


def read_json_file(path):
    with open(path) as f:
        json_input = f.read().replace('\r', '').replace('\n', '')
    try:
        return json.loads(json_input)
    except (ValueError, KeyError, TypeError) as e:
        print(f"JSON format error in file ${path}: \n ${e}")

if __name__ == "__main__":
    path = argv[1]

    config = read_json_file(path)
    training_parameters = ArchitectureConfig.parse_obj(config)
    manager = Manager(training_parameters)
    manager.execute()
