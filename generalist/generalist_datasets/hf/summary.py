from typing import Any, Callable, Dict
from datasets import load_dataset
from omegaconf import DictConfig


import torchvision.transforms as transforms
from torch.utils.data import Dataset

from generalist.data_types.helper_types import SampleBuilderMixin
from generalist.data_types.input_types import TextType
from generalist.generalist_datasets.dataset_utils import DatasetRegistry
from generalist.generalist_datasets.utils.tasks_utils import TaskInterface
from generalist.generalist_tokenizers import text_tokenizers


class SummaryTransforms:
    train = transforms.Compose([])
    val = transforms.Compose([])
    test = transforms.Compose([])

    @staticmethod
    def use_text_tokenizer(text_tokenizer: text_tokenizers.TextTokenizer, text_tokenizer_kwargs: Dict[str, Any]):
        def _to_text_type(text: str):
            text = text_tokenizer.encode_plus(text, **text_tokenizer_kwargs)
            text_type = TextType(text["input_ids"])
            other = {"attention_mask": text["attention_mask"], "token_type_ids": text["token_type_ids"]}
            return text_type, other

        return _to_text_type

    @classmethod
    def make_transforms(cls, text_tokenizer: text_tokenizers.TextTokenizer, text_tokenizer_kwargs: Dict[str, Any]):
        transform_func = transforms.Lambda(SummaryTransforms.use_text_tokenizer(text_tokenizer, text_tokenizer_kwargs))

        transform_cls = cls()
        transform_cls.train = transforms.Compose([transform_func])
        transform_cls.test = transforms.Compose([transform_func])
        transform_cls.val = transforms.Compose([transform_func])

        return transform_cls

    def __call__(self, *args, **kwargs):
        return self.train(*args, **kwargs)


class BaseSummary(SampleBuilderMixin):
    def __init__(
        self,
        path: str,
        split: str,
        text_transform: Callable = None,
        target_transform: Callable = None,
        *args,
        **kwargs,
    ) -> None:

        self._dataset = load_dataset(path, split=split)

        self.text_transform = text_transform
        self.summary_transform = text_transform

        self.sample_builder.with_task_type(TaskInterface.text_summary)

    def __len__(self):
        return len(self._dataset)


# @DatasetRegistry.register
class BillSum(BaseSummary):
    def __init__(self, *args, **kwargs):
        split = kwargs.pop("split", "ca_train")
        super().__init__(path="billsum", split=split, *args, **kwargs)

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


@DatasetRegistry.register
class XSum(BaseSummary):
    def __init__(self, *args, **kwargs):
        split = kwargs.pop("split", "train")
        super().__init__(path="xsum", split=split, *args, **kwargs)

    def __getitem__(self, idx: int | slice, *args, **kwargs):
        obj = self._dataset[idx]
        document, summary, doc_id = obj["document"], obj["summary"], obj["id"]
        text_summary = TaskInterface.text_summary(summary=summary, document=document)

        if isinstance(text_summary, tuple):
            summary, document = text_summary

        text = None
        target = None
        masks = {}

        if self.text_transform:
            text, text_other = self.text_transform(document)
            masks["data"] = text_other["attention_mask"]

        if self.summary_transform:
            target, target_other = self.summary_transform(summary)
            masks["target"] = target_other["attention_mask"]

        sample_metadata = self.sample_builder.metadata(idx=idx, dataset_name=self.__class__.__name__)
        sample = self.sample_builder(
            data=text if text is not None else document,
            target=target if target is not None else summary,
            masks=masks if masks is not {} else None,
            metadata=sample_metadata,
        )

        return sample

    @classmethod
    def from_cfg(cls, split: str, tokenizers: DictConfig, **kwargs):
        transforms = SummaryTransforms.make_transforms(
            text_tokenizer=tokenizers.text, text_tokenizer_kwargs=tokenizers.text_tokenizer_encode_kwargs
        )

        return cls(split=split, text_transform=transforms)
