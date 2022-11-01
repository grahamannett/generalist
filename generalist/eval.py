from typing import Sequence, Tuple

import torch
from omegaconf import DictConfig
from torchmetrics.functional.text.bleu import bleu_score
from torchmetrics.functional.text.rouge import rouge_score

from generalist.data_types.helper_types import Sample
from generalist.generalist_tokenizers import image_tokenizers, text_tokenizers
from generalist.models.model import GeneralistModel
from generalist.predict import ImageCaptionPrediction


def eval_summary(predictions: Sequence[str], references: Sequence[str]):
    # scores = rouge_score.compute(predictions=predictions, references=references)
    rogue_scores = rouge_score(preds=predictions, target=references)
    bleu_scores = bleu_score(preds=predictions, target=references, n_gram=1)

    return {"rogue_scores": rogue_scores, "bleu_score": bleu_scores}


class EvalMetrics:
    def __init__(self):
        self.metrics = {}

    def __call__(self, *args, **kwd):
        pass


def get_max_length(arr: torch.Tensor) -> int:
    return torch.where(arr == 0)[1][0].item()


def make_out_target_true(target: torch.Tensor, text_tokenizer: text_tokenizers.TextTokenizer, max_length: int = 32) -> str:
    max_length = get_max_length(target)
    out_target_true = text_tokenizer.decode(target[0, :max_length], skip_special_tokens=True)
    return out_target_true


def preliminary_eval(
    sample: Sample,
    text_tokenizer: text_tokenizers.TextTokenizer,
    image_tokenizer: image_tokenizers.ImageTokenizer,
    model: GeneralistModel,
    device: str,
    initial_caption: bool,
) -> Tuple[Sequence[Sequence[int]], ImageCaptionPrediction, str]:
    sample_data = sample.data
    out_target_tokens = sample.target
    sample_data_mask = sample.masks["data"]

    max_length = get_max_length(sample.target)
    out_target_true = make_out_target_true(out_target_tokens, text_tokenizer, max_length=max_length)

    caption_preder = ImageCaptionPrediction(image_tokenizer=image_tokenizer, text_tokenizer=text_tokenizer, model=model, device=device)

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


class Commands:
    eval = eval_predictions


if __name__ == "__main__":
    eval()
