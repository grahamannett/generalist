from torch.utils.data import Dataset

from typing import Any, Sequence
from generalist.generalist_tokenizers.general_tokenizer import GeneralTokenizer
from generalist.data_types.input_types import InputType
from generalist.data_types.helper_types import Sample, SampleBuilder, SampleMetaData, SampleBuilderMixin


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


class GeneralistDataset(SampleBuilderMixin, Dataset):
    shortname = None
    tokenizers = TokenizersHandler()

    def __init__(self, process_data: bool = False, process_target: bool = False, **kwargs) -> None:
        self._include_metadata = kwargs.get("include_metadata", True)

        self.process_data = process_data
        self.process_target = process_target

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

        # metadata = self.metadata.make(idx, **kwargs)
        # # metadata: SampleMetaData = kwargs.get("metadata", None)
        # # if self._include_metadata:
        # #     metadata = SampleMetaData(idx=idx, dataset_name=self.shortname)

        # return Sample(metadata=metadata)
        raise NotImplementedError("Implement this on subclass")

    def extra_metadata(self, *args, **kwargs):
        raise NotImplementedError("Implement this on subclass")
