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
        self._sample_metadata = kwargs.get("sample_metadata", True)

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
        if self._sample_metadata:
            metadata = SampleMetaData(idx=idx, dataset_name=self.shortname)

        return Sample(metadata=metadata)

    def _return_tuple(self):
        pass

    def path(self, data_type: str) -> Any:
        match data_type:
            case self.paths.image_path.data_type:
                return data_type
            case self.paths.text_path.data_type:
                return data_type

    def process_sample(self, sample: Sample, return_raw: bool = None) -> Sample:
        if self.return_raw or return_raw:
            return sample

        if self.process_sample_data:
            self.process_sample_property(sample, "data")
        if self.process_sample_target:
            self.process_sample_property(sample, "target")

        return sample

    def process_sample_property(self, sample: Sample, name: str) -> Sample:
        value = getattr(sample, name)

        match value:
            case list():
                new_data = [self._from_tokenizer(p) for p in value]
                setattr(sample, value, new_data)
            case InputType():
                setattr(sample, name, self._from_tokenizer(value))
            case _:
                raise ValueError(f"{value} is not a list or InputType")

    def _from_tokenizer(self, prop: InputType):
        breakpoint()
        return self.tokenizers[prop.data_type](prop).to(self.device).set_data_type(prop.data_type)


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
