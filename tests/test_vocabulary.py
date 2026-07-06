import os
import unittest

from pepinvent.supervised_learning.trainer.create_vocabulary import VocabularyMaker


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.data_file_path = f"{ROOT_DIR}/fixtures/test_data.csv"

        self.vocab_maker = VocabularyMaker()

    def test_vocabulary(self):
        vocabulary = self.vocab_maker.create_vocabulary(self.data_file_path, self.data_file_path)
        self.assertEqual(len(vocabulary.tokens()), 36)
