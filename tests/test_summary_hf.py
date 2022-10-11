from ensurepip import version
import unittest

from generalist.generalist_datasets.hf.summary import BillSum, BillSumTransforms
from hydra import initialize, compose
from generalist.generalist_tokenizers import text_tokenizers

from generalist.utils.utils import get_hostname


class TestBillSummary(unittest.TestCase):
    def setUp(self) -> None:

        self.text_tokenizer = text_tokenizers.TextTokenizerBert.from_pretrained("bert-base-uncased")
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name=get_hostname())
            self.text_tokenizer_kwargs = cfg.text_tokenizer

    def test_bill_summary(self):
        dataset = BillSum(
            text_transform=BillSumTransforms.get(text_tokenizer=self.text_tokenizer, text_tokenizer_kwargs=self.text_tokenizer_kwargs).train
        )
        out = dataset[0]
        self.assertIsNotNone(out.data)
