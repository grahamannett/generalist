import torch
from typing import Optional
from generalist.models.model import GeneralistModel
from generalist.data_types.input_types import ImageType

from torch.nn import functional as F


class ImageCaptionPrediction:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def make_caption(self, model: GeneralistModel, tokenized_image: ImageType, tokenized_caption):
        normal_caption = self.tokenizer.batch_decode(tokenized_caption)
        logits = model([tokenized_image, tokenized_caption])
        breakpoint()
        generated_caption = self.logits_to_sentence(logits, sequence_length=tokenized_caption.shape[-1])
        print(f"generated caption:\n==>{generated_caption}")
        print(f"actual caption:\n==>{normal_caption}")
        return {"normal": normal_caption, "generated": generated_caption}

    def generate(self, logits: torch.Tensor, max_length: int):
        # more similar to minGPT example
        # https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        for _ in range(max_length):
            pass

    def logits_to_sentence(self, logits: torch.Tensor, sequence_length: int) -> str:
        generated_caption = logits.argmax(1)[:, :sequence_length]
        generated_caption = self.tokenizer.batch_decode(generated_caption)
        return generated_caption


def top_k_top_p_filtering(
    next_token_logits: torch.FloatTensor,
    top_k: Optional[float] = None,
    top_p: Optional[float] = None,
    device: str | torch.device = "cpu",
) -> torch.FloatTensor:
    # https://sachinruk.github.io/blog/pytorch/huggingface/2021/12/28/vit-to-gpt2-encoder-decoder-model.html

    if top_k is None:
        top_k = next_token_logits.shape[-1]
    if top_p is None:
        top_p = 1.0

    p, largest_p_idx = F.softmax(next_token_logits, dim=-1).topk(top_k, dim=-1)
    cumulative_p = p.cumsum(dim=-1)
    threshold_repeated = top_p + torch.zeros((len(p), 1)).to(device)
    idx = torch.searchsorted(cumulative_p, threshold_repeated).clip(max=top_k - 1).squeeze()
    cutoffs = cumulative_p[torch.arange(len(cumulative_p)), idx]
    censored_p = (cumulative_p <= cutoffs[:, None]) * p
    renormalized_p = censored_p / censored_p.sum(dim=-1, keepdims=True)

    final_p = torch.zeros_like(next_token_logits)
    row_idx = torch.arange(len(p)).unsqueeze(1).repeat(1, top_k).to(device)
    final_p[row_idx, largest_p_idx] = renormalized_p.to(final_p.dtype)

    return final_p
