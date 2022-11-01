
import hydra
from generalist.utils.utils import get_hostname


def main():
    with hydra.initialize(version_base=None, config_path="../../config"):
        base_cfg = hydra.compose(config_name=get_hostname())

    with hydra.initialize(config_path="."):
        exp_cfg = hydra.compose(config_name="experiment")

    cfg = combine_cfgs(base_cfg, exp_cfg)

    model_save_dir = Path(cfg.model_save_dir)
    device = cfg.device
    context_length = cfg.context_length

    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    n_epochs = cfg.training.n_epochs

    model_dim = cfg.model.model_dim

    image_tokenizer = image_tokenizers.ImageTokenizer(device=device)
    text_tokenizer = text_tokenizers.TextTokenizerBert.from_pretrained("bert-base-uncased")