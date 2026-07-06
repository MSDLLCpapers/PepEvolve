import os
import shutil
import unittest

from pepinvent.supervised_learning.trainer.architecturedto import ArchitectureConfig
from pepinvent.supervised_learning.trainer.transformer_trainer import TransformerTrainer
from tests.fixtures.config import TestInputDTO, read_json_file


class TestSupervisedLearningFromCheckpoint(unittest.TestCase):

    def setUp(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        config = read_json_file(os.path.join(ROOT_DIR, 'fixtures/test_config.json'))
        self.sampling_paths = TestInputDTO(**config)



        config = ArchitectureConfig(name="test_training",
                                    training_data_path=f"{ROOT_DIR}/fixtures/test_data.csv",
                                    validation_data_path=f"{ROOT_DIR}/fixtures/test_data.csv",
                                    save_directory=self.sampling_paths.test_folder,
                                    num_epoch=3, starting_epoch=3)
        self._trainer = TransformerTrainer(config)


    def tearDown(self):
        shutil.rmtree(self.sampling_paths)


    def test_training_from_checkpoint(self):
        self._trainer.execute()
