from typing import Any
import torch


from generalist.generalist_tokenizers.general_tokenizer import GeneralTokenizer
from generalist.data_types.input_types import GeneralizedTensor, TextType
from transformers import GPT2Model, GPT2PreTrainedModel, XLNetTokenizer, BertTokenizer, PreTrainedTokenizer


class TextTokenizer(GeneralTokenizer):
    """
    Text is encoded via SentencePiece (Kudo and Richardson, 2018) with 32000 subwords into the integer range [0, 32000).
    """

    data_type = TextType.data_type

    def __init__(
        self,
        padding: bool = True,
        tokenizer_class: str = BertTokenizer,
        pretrained_model_or_path: str = "bert-base-uncased",
        # tokenizer_class=XLNetTokenizer,
        # pretrained_model_or_path: str = "xlnet-base-cased",
        model_max_length: int = 1024,
        padding_side: str = "right",
        truncation: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.tokenizer = tokenizer_class.from_pretrained(
            pretrained_model_or_path,
            padding=padding,
            model_max_length=model_max_length,
            padding_side=padding_side,
            do_lower=True,
        )
        self.return_tensors = "pt"
        self.model_max_length = model_max_length
        self.truncation = truncation

        #
        # self.max_length = (self.max_length,)
        # self.pad_to_max_length = (True,)
        # self.return_attention_mask = (True,)
        # self.return_token_type_ids = (False,)

    def __call__(self, sample: TextType | str, **kwargs) -> torch.Tensor:
        text = sample.data if isinstance(sample, TextType) else sample
        encoded = self.encode(text, **kwargs)

        input_ids = TextType(encoded["input_ids"])
        input_ids.set_data_type(self.data_type)
        # superflous text data
        input_ids.attention_mask = encoded["attention_mask"]
        input_ids.token_type_ids = encoded["token_type_ids"]
        return input_ids

    def encode(self, text: str, **kwargs) -> torch.Tensor:
        max_length = kwargs.pop("max_length", self.model_max_length)
        return_tensors = kwargs.pop("return_tensors", self.return_tensors)
        truncation = kwargs.pop("truncation", self.truncation)
        pad_to_max_length = kwargs.pop("pad_to_max_length", True)

        encoded = self.tokenizer(
            text,
            return_tensors=return_tensors,
            truncation=truncation,
            pad_to_max_length=pad_to_max_length,
            max_length=max_length,
            **kwargs,
        )

        return encoded

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


def TextTokenizerPretrained(tokenizer_class: PreTrainedTokenizer | str, pretrained_name_or_model: str, **kwargs):
    """

    issue with saving this object which i am saving because it is slightly easier to make eval work
    use this like
    text_tokenizer = TextTokenizerPretrained("BertTokenizer", pretrained_name_or_model="bert-base-uncased", device=device)
    """
    if isinstance(tokenizer_class, str):
        exec(f"from transformers import {tokenizer_class}")
        tokenizer_class = locals().get(tokenizer_class, None)

    class DynamicClass(tokenizer_class):
        data_type = TextType.data_type

        def __call__(self, *args, **kwargs):
            # do this otherwise have to worry about tensor type
            if _original := kwargs.pop("original", False):
                return super().__call__(*args, **kwargs)

            if "return_tensors" not in kwargs:
                kwargs["return_tensors"] = "pt"

            encoded = super().__call__(*args, **kwargs)
            input_ids = TextType(encoded["input_ids"])
            # input_ids.set_data_type(self.data_type)
            return input_ids

        def __repr__(self) -> str:
            return (
                f"{self.__class__.__name__}(name_or_path='{self.name_or_path}',"
                f" vocab_size={self.vocab_size}, model_max_len={self.model_max_length}, is_fast={self.is_fast},"
                f" padding_side='{self.padding_side}', truncation_side='{self.truncation_side}',"
                f" special_tokens={self.special_tokens_map_extended})"
            )

    DynamicClass.__name__ = f"TextTokenizer_{tokenizer_class.__name__}"
    instance = DynamicClass.from_pretrained(pretrained_name_or_model)
    if "device" in kwargs:
        instance.device = kwargs["device"]
    return instance


class TextTokenizerBert(BertTokenizer):
    data_type = TextType.data_type

    def __call__(self, *args, **kwargs):
        return self.__call__default(*args, **kwargs)
        # return self.encode_to_type(*args, **kwargs)

    def encode_to_type(self, *args, **kwargs):
        if _original := kwargs.pop("original", False):
            return super().__call__(*args, **kwargs)

        if "return_tensors" not in kwargs:
            kwargs["return_tensors"] = "pt"

        encoded = super().__call__(*args, **kwargs)
        tokenized_text = TextType(encoded["input_ids"])

        if "return_attention_mask" in kwargs:
            tokenized_text.attention_mask = encoded["attention_mask"]

        return tokenized_text

    def original_encode(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    __call__default = original_encode


if __name__ == "__main__":
    # test = TextTokenizerPretrained(XLNetTokenizer, pretrained_name_or_model="xlnet-base-cased")
    test = TextTokenizerPretrained("XLNetTokenizer", pretrained_name_or_model="xlnet-base-cased")
    out = test("hekahsda how are you??")
    # text_tokenizer = TextTokenizerPretrained_._from_pretrained()
    breakpoint()
    # breakpoint()
