import torch
from typing import Optional
from generalist.generalist_tokenizers.image_tokenizers import ImageTokenizer
from generalist.generalist_tokenizers.text_tokenizers import TextTokenizer
from generalist.models.embedding_model import EmbeddingModel
from generalist.models.model import GeneralistModel
from generalist.data_types.input_types import ImageType, TextType, TextTypeTensor

from torch.nn import functional as F


class ImageCaptionPrediction:
    def __init__(
        self,
        image_tokenizer: ImageTokenizer,
        text_tokenizer: TextTokenizer,
        embedding_model: EmbeddingModel,
        model: GeneralistModel,
        device: str = None,
    ) -> None:
        self.image_tokenizer = image_tokenizer
        self.text_tokenizer = text_tokenizer
        self.embedding_model = embedding_model
        self.model = model
        self.device: str = device

    def make_caption(
        self,
        # embedding_model: EmbeddingModel,
        # model: GeneralistModel,
        data: torch.Tensor,
        max_length: int = 32,
        # tokenized_caption: TextType,
        use_caption: bool = True,
        **kwargs,
    ):

        # target_list = [self.text_tokenizer.cls_token_id]
        embedded_data = self.embedding_model([data])

        target_list_top_k_p = [self.text_tokenizer.cls_token_id]
        target_list_top_p = [self.text_tokenizer.cls_token_id]
        target_list_top_k = [self.text_tokenizer.cls_token_id]
        target_list_top_a = [self.text_tokenizer.cls_token_id]

        target_list = target_list_top_k_p

        for i in range(max_length):

            # tokenized_target = TextType(target_list).to(int).to(image.device)
            tokenized_target = TextTypeTensor(target_list).to(int).to(data.device)
            # embedded_tgt = embedding_model([tokenized_target]) if use_caption else None
            embedded_tgt = self.embedding_model(tokenized_target)
            logits = self.model(embedded_data, embedded_tgt)

            # token_pred = top_k_top_p_filtering(logits[:, -1, :], device=logits.device).argmax().item()

            token_pred_top_k_p = top_k_top_p_filtering(logits[:, -1, :], device=logits.device).argmax().item()
            token_pred_top_a = top_a(logits[:, -1]).argmax().item()
            token_pred_top_k = top_k(logits[:, -1]).argmax().item()
            token_pred_top_p = top_p(logits[:, -1]).argmax().item()

            token_pred = token_pred_top_k_p

            # token_pred = top_a(logits)
            # target_list.append(token_pred)

            target_list_top_k_p.append(token_pred_top_k_p)
            target_list_top_p.append(token_pred_top_p)
            target_list_top_k.append(token_pred_top_k)
            target_list_top_a.append(token_pred_top_a)

            # breakpoint()

            if token_pred == self.text_tokenizer.sep_token_id:
                break

        # generated_caption = self.text_tokenizer.decode(target_list)
        # normal_caption = self.text_tokenizer.batch_decode(tokenized_caption)[0]
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


# from x_transformers
from math import ceil
import torch
from torch import nn
import torch.nn.functional as F


def exists(val):
    return val is not None


# nucleus


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


# topk


def top_k(logits, thres=0.9):
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


# top_a


def top_a(logits, min_p_pow=2.0, min_p_ratio=0.02):
    probs = F.softmax(logits, dim=-1)
    limit = torch.pow(torch.max(probs), min_p_pow) * min_p_ratio
    logits[probs < limit] = -float("Inf")
    logits[probs >= limit] = 1
    return logits


@torch.no_grad()
def generate(
    encoder,
    decoder,
    text_tokenizer,
    image,
    start_tokens,
    seq_len,
    eos_token=None,
    temperature=1.0,
    filter_logits_fn=top_k,
    filter_thres=0.9,
    min_p_pow=2.0,
    min_p_ratio=0.02,
    max_seq_len=16,
    ignore_index=-100,
    pad_value=0,
    **kwargs,
):
    device = start_tokens.device
    was_training = encoder.training
    num_dims = len(start_tokens.shape)

    # breakpoint()
    # if num_dims == 1:
    #     start_tokens = start_tokens[None, :]

    if start_tokens.ndim == 1:
        start_tokens = start_tokens.unsqueeze(0)
    b, t = start_tokens.shape

    # self.net.eval()
    encoder.eval()
    decoder.eval()
    out = start_tokens
    mask = kwargs.pop("mask", None)

    encoded = encoder(image, return_embeddings=True)

    for _ in range(seq_len):
        x = out[:, -max_seq_len:]

        # logits = self.net(x, mask=mask, **kwargs)[:, -1, :]
        logits = decoder(x, context=encoded)
        last = logits[:, -1]

        filtered_logits = filter_logits_fn(last, thres=filter_thres)
        probs = F.softmax(filtered_logits / temperature, dim=-1)

        sample = torch.multinomial(probs, num_samples=1)

        #     # if filter_logits_fn in {top_k, top_p}:
        #     #     filtered_logits = filter_logits_fn(logits, thres=filter_thres)
        #     #     probs = F.softmax(filtered_logits / temperature, dim=-1)

        #     # elif filter_logits_fn is top_a:
        #     #     filtered_logits = filter_logits_fn(logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio)
        #     #     probs = F.softmax(filtered_logits / temperature, dim=-1)

        #     breakpoint()

        #     sample = torch.multinomial(probs, 1)

        out = torch.cat((out, sample), dim=-1)

        if exists(eos_token) and (sample == eos_token).any():
            break

        # mask = F.pad(mask, (0, 1), value=True)

    #     if exists(eos_token):
    #         is_eos_tokens = out == eos_token

    #         if is_eos_tokens.any(dim=-1).all():
    #             # mask out everything after the eos tokens
    #             shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
    #             mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
    #             out = out.masked_fill(mask, pad_value)
    #             break

    # out = out[:, t:]

    # if num_dims == 1:
    #     out = out.squeeze(0)

    # self.net.train(was_training)
    return out
