from typing import Any, Dict
import torchvision.transforms as transforms
from generalist.generalist_tokenizers import text_tokenizers


class TextTransforms:
    train = transforms.Compose([])
    val = transforms.Compose([])

    @staticmethod
    def use_text_tokenizer(text_tokenizer: text_tokenizers.TextTokenizer, text_tokenizer_kwargs: Dict[str, Any]):
        def _to_text_type(text: str):
            text = text_tokenizer.encode_plus(text, **text_tokenizer_kwargs)
            return TextType(text["input_ids"]), {"attention_mask": text["attention_mask"]}

        return _to_text_type

    @classmethod
    def get(cls, text_tokenizer: text_tokenizers.TextTokenizer, text_tokenizer_kwargs: Dict[str, Any]):
        _transforms = cls()
        _transforms.train.transforms.append(transforms.Lambda(TextTransforms.use_text_tokenizer(text_tokenizer, text_tokenizer_kwargs)))
        return _transforms

    def __call__(self, *args, **kwargs):
        return self.train(*args, **kwargs)
