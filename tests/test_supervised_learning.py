import os
import unittest

from pepinvent.supervised_learning.trainer.architecturedto import ArchitectureConfig
from pepinvent.supervised_learning.trainer.transformer_trainer import TransformerTrainer


class TestSupervisedLearning(unittest.TestCase):

    def setUp(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        config = ArchitectureConfig(name="test_training",
                                    training_data_path=f"{ROOT_DIR}/fixtures/test_data.csv",
                                    validation_data_path=f"{ROOT_DIR}/fixtures/test_data.csv",
                                    save_directory="/tmp/pepinvent_testing",
                                    num_epoch=2, starting_epoch=1, batch_size=16)
        self._trainer = TransformerTrainer(config)


    def tearDown(self):
        pass


    def test_training(self):
        self._trainer.execute()
