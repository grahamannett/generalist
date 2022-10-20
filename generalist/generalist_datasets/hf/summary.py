from typing import Any, Callable, Dict
from datasets import load_dataset


import torchvision.transforms as transforms
from torch.utils.data import Dataset

from generalist.data_types.helper_types import SampleBuilderMixin
from generalist.data_types.input_types import TextType
from generalist.generalist_datasets.utils.tasks_utils import TaskInterface
from generalist.generalist_tokenizers import text_tokenizers


class SummaryTransforms:
    train = transforms.Compose([])
    val = transforms.Compose([])

    @staticmethod
    def use_text_tokenizer(text_tokenizer: text_tokenizers.TextTokenizer, text_tokenizer_kwargs: Dict[str, Any]):
        def _to_text_type(text: str):
            text = text_tokenizer.encode_plus(text, **text_tokenizer_kwargs)
            # return TextType(text["input_ids"]), {"attention_mask": text["attention_mask"], "task_type":
            return TextType(text["input_ids"]), {"attention_mask": text["attention_mask"]}

        return _to_text_type

    @classmethod
    def make_transforms(cls, text_tokenizer: text_tokenizers.TextTokenizer, text_tokenizer_kwargs: Dict[str, Any]):
        _transforms = cls()
        _transforms.train.transforms.append(transforms.Lambda(SummaryTransforms.use_text_tokenizer(text_tokenizer, text_tokenizer_kwargs)))
        return _transforms

    def __call__(self, *args, **kwargs):
        return self.train(*args, **kwargs)


class BaseSummary(SampleBuilderMixin):
    def __init__(
        self,
        path: str,
        split: str,
        text_transform: Callable = SummaryTransforms.train,
        *args,
        **kwargs,
    ) -> None:

        self._dataset = load_dataset(path, split=split)

        self.text_transform = text_transform
        self.summary_transform = text_transform

        self.sample_builder.with_task_type(TaskInterface.text_summary)

    def __len__(self):
        return len(self._dataset)

    def _to_sample(self, idx: int, document, summary, other):

        if self.text_transform:
            text, text_other = self.text_transform(raw_text)
        if self.summary_transform:
            target, target_other = self.summary_transform(target)

        sample_metadata = self.sample_builder.metadata(idx=idx, dataset_name=self.__class__.__name__)
        masks = {"data": text_other["attention_mask"], "target": target_other["attention_mask"]}

        sample = self.sample_builder(data=text, target=target, masks=masks, metadata=sample_metadata)

        return sample


class BillSum(BaseSummary):
    def __init__(self, *args, **kwargs):
        super().__init__(path="billsum", split="ca_test", *args, **kwargs)

    def __getitem__(self, idx: int | slice, *args, **kwargs):
        raw_text, summary, title = self._dataset[idx]["text"], self._dataset[idx]["summary"], self._dataset[idx]["title"]
        target = TaskInterface.text_summary(summary=summary)

        if self.text_transform:
            text, text_other = self.text_transform(raw_text)
        if self.summary_transform:
            target, target_other = self.summary_transform(target)

        sample_metadata = self.sample_builder.metadata(idx=idx, dataset_name=self.__class__.__name__)
        masks = {"data": text_other["attention_mask"], "target": target_other["attention_mask"]}

        sample = self.sample_builder(data=text, target=target, masks=masks, metadata=sample_metadata)

        return sample


class XSum(BaseSummary):
    def __init__(self, *args, **kwargs):
        super().__init__(path="xsum", split="train", *args, **kwargs)

    def __getitem__(self, idx: int | slice, *args, **kwargs):
        obj = self._dataset[idx]
        document, summary, doc_id = obj["document"], obj["summary"], obj["id"]
        target = TaskInterface.text_summary(summary=summary)

        if self.text_transform:
            text, text_other = self.text_transform(document)
        if self.summary_transform:
            target, target_other = self.summary_transform(target)

        sample_metadata = self.sample_builder.metadata(idx=idx, dataset_name=self.__class__.__name__)
        masks = {"data": text_other["attention_mask"], "target": target_other["attention_mask"]}

        sample = self.sample_builder(data=text, target=target, masks=masks, metadata=sample_metadata)

        return sample
