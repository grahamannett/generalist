from hydra import initialize, compose
from generalist.generalist_tokenizers import text_tokenizers

from generalist.utils.utils import get_hostname


def get_cfg(self):
    if not hasattr(self, "cfg"):
        with initialize(config_path="../config", version_base=None):
            self.cfg = compose(config_name=get_hostname())
            return self.cfg
    else:
        return self.cfg


class CfgTestMixin:
    def setUp(self) -> None:
        self.cfg = get_cfg(self)


class TextTestMixin:
    def setUp(self) -> None:
        self.cfg = get_cfg(self)
        self.text_tokenizer = text_tokenizers.TextTokenizerBert.from_pretrained("bert-base-uncased")
        self.text_tokenizer_kwargs = self.cfg.text_tokenizer
