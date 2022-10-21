from typing import Any, Callable, Dict, Sequence
import torch
import hydra
from generalist.data_types.helper_types import Sample
from generalist.generalist_datasets.coco.coco import (
    CocoCaption,
    CocoCaptionTargetTranform,
    CocoFilepaths,
    CocoImageTransforms,
)
from generalist.generalist_datasets.coco.eval import CocoEval
from generalist.generalist_tokenizers import image_tokenizers, text_tokenizers
from generalist.models.embedding_model import EmbeddingModel
from generalist.models.model import GeneralistModel

from generalist.utils.utils import get_hostname
from omegaconf import DictConfig

from generalist.predict import ImageCaptionPrediction


def get_max_length(arr: torch.Tensor) -> int:
    return torch.where(arr == 0)[1][0].item()


def make_out_target_true(target: torch.Tensor, text_tokenizer: text_tokenizers.TextTokenizer, max_length: int = 32) -> str:
    max_length = get_max_length(target)
    out_target_true = text_tokenizer.decode(target[0, :max_length])
    return out_target_true


def preliminary_eval(
    sample: Sample,
    text_tokenizer: text_tokenizers.TextTokenizer,
    image_tokenizer: image_tokenizers.ImageTokenizer,
    embedding_model: EmbeddingModel,
    model: GeneralistModel,
    device: str,
    initial_caption: bool,
) -> Sequence[Sequence[int]]:
    sample_data = sample.data
    out_target_tokens = sample.target

    max_length = get_max_length(sample.target)
    out_target_true = make_out_target_true(out_target_tokens, text_tokenizer, max_length=max_length)

    caption_preder = ImageCaptionPrediction(
        image_tokenizer=image_tokenizer, text_tokenizer=text_tokenizer, embedding_model=embedding_model, model=model, device=device
    )

    # tokenized_caption = out_target_tokens.to(device)
    generated_captions = []

    if initial_caption:
        initial_caption = caption_preder.generate_output(
            data=sample_data,
            max_length=max_length,
        )
        generated_captions.append(initial_caption)
    return generated_captions, caption_preder, out_target_true


def eval_predictions(cfg: DictConfig):
    pass

    # def on_job_end(self, config: DictConfig, **kwargs: Any) -> None:
    #     print(f"Job ended,uploading...")
    #     # uploading...

def main():
    pass


class Commands:
    eval = eval_predictions


if __name__ == "__main__":
    eval()
