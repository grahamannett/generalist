from typing import Any, Callable, Dict
from datasets import load_dataset


import torchvision.transforms as transforms
from torch.utils.data import Dataset

from generalist.data_types.helper_types import SampleBuilderMixin
from generalist.data_types.input_types import TextType
from generalist.generalist_datasets.utils.tasks_utils import TaskInterface
from generalist.generalist_tokenizers import text_tokenizers


class BillSumTransforms:
    train = transforms.Compose([])
    val = transforms.Compose([])

    @staticmethod
    def use_text_tokenizer(text_tokenizer: text_tokenizers.TextTokenizer, text_tokenizer_kwargs: Dict[str, Any]):
        def _to_text_type(text: str):
            text = text_tokenizer.encode_plus(text, **text_tokenizer_kwargs)
            return TextType(text["input_ids"]), text["attention_mask"]

        return _to_text_type

    @classmethod
    def get(cls, text_tokenizer: text_tokenizers.TextTokenizer, text_tokenizer_kwargs: Dict[str, Any]):
        _transforms = cls()
        _transforms.train.transforms.append(transforms.Lambda(BillSumTransforms.use_text_tokenizer(text_tokenizer, text_tokenizer_kwargs)))
        return _transforms

    def __call__(self, *args, **kwargs):
        return self.train(*args, **kwargs)


class BillSum(SampleBuilderMixin):
    def __init__(
        self,
        split: str = "ca_test",
        text_transform: Callable = BillSumTransforms.train,
        # summary_transform: Callable = BillSumTransforms.train,
        *args,
        **kwargs,
    ) -> None:
        # self._dataset = load_dataset("billsum")
        self._dataset = load_dataset("billsum", split=split)

        self.text_transform = text_transform
        # self.summary_transform = summary_transform
        self.summary_transform = text_transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int | slice, *args, **kwargs):
        raw_text, summary, title = self._dataset[idx]["text"], self._dataset[idx]["summary"], self._dataset[idx]["title"]
        target = TaskInterface.text_summary(summary=summary)

        if self.text_transform:
            text, text_mask = self.text_transform(raw_text)
        if self.summary_transform:
            target, target_mask = self.summary_transform(target)

        sample_metadata = self.sample_builder.metadata(idx=idx, dataset_name=self.__class__.__name__)
        masks = {"data": text_mask, "target": target_mask}

        sample = self.sample_builder(data=text, target=target, masks=masks, metadata=sample_metadata)

        return sample
