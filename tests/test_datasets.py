import unittest

from generalist.generalist_datasets.dataset_utils import GeneralistDataset


class TestDatasets(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_summarization_dataset(self):
        train_dataset = GeneralistDataset.get("hf_summarization")
        self.assertIsNotNone(train_dataset)

        test_dataset = GeneralistDataset["hf_summarization"](split="test")
        self.assertIsNotNone(test_dataset)

        test_batch = test_dataset[0]
        self.assertIsNotNone(test_batch)
