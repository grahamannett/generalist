from torch.utils.data import Dataset

from typing import Any, Sequence
from generalist.generalist_tokenizers.general_tokenizer import GeneralTokenizer
from generalist.data_types.input_types import InputType
from generalist.data_types.helper_types import Sample, SampleMetaData


class TokenizersHandler:
    _tokenizers: dict[str, Any] = {}

    def __setitem__(self, key, val):
        self._tokenizers[key] = val
        setattr(self, key, val)

    def __getitem__(self, key):
        return self._tokenizers[key]

    def __repr__(self):
        _tokenizers_str = "\n".join([f"\t{key}:\n\t\t{val}" for key, val in self._tokenizers.items()])
        return f"{self.__class__.__name__}\n{_tokenizers_str}"


class GeneralistDataset(Dataset):
    shortname = None
    tokenizers = TokenizersHandler()

    def __init__(self, device: str, return_raw: bool = False, **kwargs) -> None:
        self._include_metadata = kwargs.get("include_metadata", True)

        self.return_raw = return_raw
        self.process_sample_data = kwargs.get("process_sample_data", True)
        self.process_sample_target = kwargs.get("process_sample_target", True)

        self.device = device

    @classmethod
    def use_tokenizers(cls, tokenizers: Sequence[GeneralTokenizer], *args, **kwargs) -> None:
        if isinstance(tokenizers, GeneralTokenizer):
            tokenizers = [GeneralTokenizer]
        if args:
            tokenizers.extend(args)

        for tokenizer in tokenizers:
            cls.tokenizers[tokenizer.data_type] = tokenizer

        # cls.tokenizers = {tok.data_type: tok for tok in tokenizers}

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        metadata: SampleMetaData = kwargs.get("metadata", None)
        if self._include_metadata:
            metadata = SampleMetaData(idx=idx, dataset_name=self.shortname)

        return Sample(metadata=metadata)

    def extra_metadata(self, *args, **kwargs):
        raise NotImplementedError("Implement this on subclass")


class ChainedGenearlistDataset(Dataset):
    def __init__(self, datasets: Sequence[GeneralistDataset], sample_weights: Sequence[float], **kwargs) -> None:
        super().__init__(**kwargs)
        self._datasets = datasets
        self.sample_weights = sample_weights

        self._lengths = [len(dataset) for dataset in self._datasets]
        self._lengths_idx = [sum(self._lengths[:i]) for i in range(len(self._lengths))]

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, index: int) -> Sample:
        dataset_idx = [_ for _ in self._lengths_idx if _ <= index].pop()
        return self._datasets[dataset_idx].__getitem__(index - self._lengths_idx[dataset_idx])


class CombinedDataset(GeneralistDataset):
    def __init__(self, datasets: Sequence[GeneralistDataset], batch_same: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._datasets = datasets

        self.batch_same = batch_same

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self._datasets])

    def __getitem__(self, index) -> Sample:
        dataset_idx = [_ for _ in self._lengths_idx if _ <= index].pop()
        return self._datasets[dataset_idx].__getitem__(index - self._lengths_idx[dataset_idx])
