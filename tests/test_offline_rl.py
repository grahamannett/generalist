import unittest

from generalist.generalist_datasets.offline_rl.hf_offline_rl import GymReplay

from tests.utils import CfgTestMixin, TextTestMixin


class TestOfflineRL(CfgTestMixin, TextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_dataset(self):
        dataset = GymReplay()
        out = dataset[0]
        breakpoint()
        # self.assertIsNone(dataset[0])

    def test_embeddings(self):
        pass
