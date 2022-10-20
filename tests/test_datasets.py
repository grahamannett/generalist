import unittest

# from generalist.generalist_datasets.base import GeneralistDataset
from generalist.generalist_datasets.hf.hf_datasets import LanguageModelingDataset


class TestDatasets(unittest.TestCase):
    def setUp(self) -> None:
        # return super().setUp()
        return

    # def test_summarization_dataset(self):
    #     train_dataset = GeneralistDataset.get("hf_summarization")
    #     self.assertIsNotNone(train_dataset)

    #     test_dataset = GeneralistDataset["hf_summarization"](split="test")
    #     self.assertIsNotNone(test_dataset)

    #     test_batch = test_dataset[0]
    #     self.assertIsNotNone(test_batch)

    #     breakpoint()
    def test_hf_wikitext(self):
        dataset = LanguageModelingDataset()
        out = dataset[0]
        # dataset = load_dataset("wikitext", "")[split]
