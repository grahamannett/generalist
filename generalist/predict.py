import torch
from typing import Optional
from generalist.generalist_tokenizers.image_tokenizer import ImageTokenizer
from generalist.generalist_tokenizers.text_tokenizer import TextTokenizer
from generalist.models.embedding_model import EmbeddingModel
from generalist.models.model import GeneralistModel
from generalist.data_types.input_types import ImageType, TextType

from torch.nn import functional as F


class ImageCaptionPrediction:
    def __init__(self, image_tokenizer: ImageTokenizer, text_tokenizer: TextTokenizer) -> None:
        self.image_tokenizer = image_tokenizer
        self.text_tokenizer = text_tokenizer

    def make_caption(
        self,
        embedding_model: EmbeddingModel,
        model: GeneralistModel,
        tokenized_image: ImageType,
        tokenized_caption: TextType,
    ):

        # breakpoint()
        if tokenized_caption.ndim == 1:
            tokenized_caption = tokenized_caption.unsqueeze(0)

        target_list = [self.text_tokenizer.cls_token_id]

        embedded_image = embedding_model([tokenized_image])

        for i in range(tokenized_caption.shape[-1]):
            tokenized_target = TextType(target_list).to(int).to(tokenized_image.device)
            embedded_target = embedding_model([tokenized_target])
            logits = model(embedded_image, embedded_target)

            token_pred = top_k_top_p_filtering(logits[:, -1, :], device=logits.device).argmax().item()
            target_list.append(token_pred)

            if token_pred == self.text_tokenizer.sep_token_id:
                break

        generated_caption = self.text_tokenizer.decode(target_list)
        normal_caption = self.text_tokenizer.batch_decode(tokenized_caption)[0]

        # print(f"generated caption:\n==>{generated_caption}")
        # print(f"actual caption:\n==>{normal_caption}")

        # return {"normal": normal_caption, "generated": generated_caption}
        return target_list

    def generate(self, logits: torch.Tensor, max_length: int):
        # more similar to minGPT example
        # https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        for _ in range(max_length):
            pass


def simple_next_token_pred(next_token_logits: torch.Tensor):
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    token_pred = logits.argmax(-1).item()
    return token_pred


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
