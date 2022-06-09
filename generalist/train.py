import argparse
from pathlib import Path
from generalist.models.gpt_fix import TransformerDecoder


import torch
from torch.utils.data import DataLoader
from transformers import AdamW, GPT2Model
from tqdm import tqdm

from config import AOKVQA_DIR, FEATURES_DIR, LOG_DIR, device
from generalist.generalist_datasets.aokvqa.aokvqa import AokvqaDataset


from generalist.generalist_tokenizers.text_embedding import TextEmbedding, TextEmbedding2

from generalist.models.model import GeneralistModel


def train():

    image = torch.rand(32, 3, 224, 224)

    tem, loading_info = TextEmbedding2.from_pretrained(
        pretrained_model_name_or_path="gpt2",
        output_loading_info=True,
    )
    breakpoint()
    model = GeneralistModel()

    # image_tokenized = image_tokenizer(image)

    # image_embedding = image_embedder(image_tokenized)
    # transformer = TransformerDecoder.from_pretrained("gpt2")
    # breakpoint()

    # breakpoint()
    # text_tokenizer = TextTokenizer()
    # text_embedder = TextEmbedding()
    # text_embedder.use_pretrained()
    text = "Hello World"
    # text_tokenized = text_tokenizer(text)
    # position_ids = torch.arange(0, text_tokenized["input_ids"].shape[-1], dtype=torch.long)

    # text_tokenized["position_ids"] = position_ids

    # outputs = model.decoder(**text_tokenized)
    # model = GPT2Model.from_pretrained("gpt2")
    # out = model(**text_tokenized)

    # model2 = GPT2Model_Fix.from_pretrained("gpt2")
    # out2 = model2(**text_tokenized)

    input_data = {
        "data": text,
        "type": "text",
    }
    out2 = model(input_data)
    breakpoint()

    # text_outputs =

    # out =
    # model = load_model(cfg, cfg.pretrained_model)
    # model.to(device)
    # optimizer = AdamW(model.parameters(), lr=cfg.lr)
    # dataset = load_data_super(cfg, "train")

    # train_dataloader = DataLoader(dataset, batch_size=cfg.bs, shuffle=True, drop_last=True)

    # for epoch in range(1, cfg.epochs):
    #     model = train_step(
    #         epoch=epoch, model=model, dataloader=train_dataloader, prefix_length=cfg.prefix_length
    #     )


def train_step(epoch, model, dataloader, prefix_length):
    print(f"on epoch=>{epoch}")
    model.train()
    loss_func = torch.nn.CrossEntropyLoss()

    for batch in tqdm(dataloader):
        prefix, input_tokens, prompt_len, target_len = batch
        breakpoint()
        model.zero_grad()

        # outputs = model(batch["inputs"])

        # target_logits = [
        #     l[s:e]
        #     for l, s, e in zip(
        #         outputs.logits, prefix_length + prompt_len - 1, prefix_length + prompt_len + target_len
        #     )
        # ]

        # target_tokens = [t[s:e] for t, s, e in zip(input_tokens, prompt_len, prompt_len + target_len + 1)]

        # loss = loss_func(torch.cat(target_logits), torch.cat(target_tokens))
        # breakpoint()


def compute_step(
    model, prefix, input_tokens, prefix_len, prompt_len, target_len, metrics=None, tokenizer=None
):

    outputs = model(prefix, input_tokens)

    ## Compute loss (comparing [target, eos] indices)

    target_logits = [
        l[s:e]
        for l, s, e in zip(outputs.logits, prefix_len + prompt_len - 1, prefix_len + prompt_len + target_len)
    ]

    target_tokens = [t[s:e] for t, s, e in zip(input_tokens, prompt_len, prompt_len + target_len + 1)]

    loss = F.cross_entropy(torch.cat(target_logits), torch.cat(target_tokens))

    ## Compute metrics (generated text vs target text)
    if metrics is not None:
        assert tokenizer is not None

        # All tokens after prompt
        generated_tokens = [
            list(l[s:-1].argmax(dim=1)) for l, s in zip(outputs.logits, prefix_len + prompt_len - 1)
        ]

        # Remove tokens at or after eos_token
        generated_tokens = [
            gen_t[: gen_t.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in gen_t else gen_t
            for gen_t in generated_tokens
        ]
        target_tokens = [tt[:-1] for tt in target_tokens]

        generated_text = [tokenizer.decode(gen_t) for gen_t in generated_tokens]
        target_text = [[tokenizer.decode(target_t)] for target_t in target_tokens]

        if "bleu" in metrics:
            metrics["bleu"].add_batch(predictions=generated_text, references=target_text)
        if "meteor" in metrics:
            metrics["meteor"].add_batch(predictions=generated_text, references=target_text)
        if "f1" in metrics:
            for gen, target in zip(generated_text, target_text):
                metrics["f1"].append(compute_f1(target[0], gen))
        if "exact" in metrics:
            for gen, target in zip(generated_text, target_text):
                metrics["exact"].append(compute_exact(target[0], gen))

    return loss


if __name__ == "__main__":
    # cfg = parse_args()
    # train(cfg)
    train()
