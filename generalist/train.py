import logging
import random
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig
from rich import print
from torch.utils.data import DataLoader
from torchvision import transforms

from generalist.data_types.helper_types import Sample
from generalist.data_types.input_types import ImageType, TextType
from generalist.generalist_datasets import AokvqaDataset, GeneralistDataset, MNISTDataset
from generalist.generalist_datasets.coco.coco import (
    CocoCaption,
    CocoCaptionTargetTranform,
    CocoDetection,
    CocoFilepaths,
    CocoRegionTargetTransform,
    CocoImageTransforms,
)
from generalist.generalist_datasets.coco.eval import CocoEval
from generalist.generalist_datasets.hf.summary import BillSum, BillSumTransforms
from generalist.generalist_datasets.utils.data_collate import collate_func_helper
from generalist.generalist_datasets.utils.tasks_utils import TaskInterface
from generalist.generalist_tokenizers import image_tokenizers, text_tokenizers
from generalist.models.model import EmbeddingModel, GeneralistModel
from generalist.models.output_model import GeneralOutput
from generalist.predict import ImageCaptionPrediction
from generalist.utils.display import GeneralistDisplay
from generalist.utils.utils import get_hostname, save_checkpoint

log = logging.getLogger(__name__)


def train_step(embedding_model, genearlist_model, dataloader):
    pass


@hydra.main(config_path=f"../config", config_name=get_hostname(), version_base=None)
def train(cfg: DictConfig):
    model_save_dir = Path(cfg.model_save_dir)
    display_flag = cfg.display.display_flag
    device = cfg.device
    context_length = cfg.context_length

    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    n_epochs = cfg.training.n_epochs

    model_dim = cfg.model.model_dim

    image_tokenizer = image_tokenizers.ImageTokenizer(device=device)
    text_tokenizer = text_tokenizers.TextTokenizerBert.from_pretrained("bert-base-uncased")

    embedding_model = EmbeddingModel(model_dim=model_dim)
    # output_model = GeneralClassificationOutput(model_dim=model_dim, num_classes=10, reduce_type="cls")
    output_model = GeneralOutput(model_dim=model_dim, output_dim=text_tokenizer.vocab_size)
    model = GeneralistModel(output_model=output_model, **cfg.model).to(device)

    start_tokens = torch.Tensor([text_tokenizer.cls_token_id]).to(device).to(int)

    embedding_model.to(device)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_model.parameters()},
            {"params": model.parameters()},
        ],
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    tokenizers = [image_tokenizer, text_tokenizer]

    text_tokenizer_kwargs = cfg.text_tokenizer

    coco_filepaths = CocoFilepaths(base_dir=cfg.coco_dir, split="train")

    coco_caption = CocoCaption(
        root=coco_filepaths.images_root,
        annFile=coco_filepaths.captions_filepath,
        transform=CocoImageTransforms.train,
        target_transform=CocoCaptionTargetTranform.get(text_tokenizer=text_tokenizer, text_tokenizer_kwargs=text_tokenizer_kwargs).train,
    )

    summary_dataset = BillSum(
        text_transform=BillSumTransforms.get(text_tokenizer=text_tokenizer, text_tokenizer_kwargs=text_tokenizer_kwargs).train,
    )
    out = summary_dataset[0]
    # out = summary_dataset[0]

    dataset = coco_caption
    sample = coco_caption[0]

    # eval example

    sample_data = sample.data
    out_target_tokens = sample.target

    _max_length = torch.where(sample.target == 0)[1][0].item()
    out_target_true = text_tokenizer.decode(out_target_tokens[0, :_max_length])

    caption_preder = ImageCaptionPrediction(
        image_tokenizer=image_tokenizer, text_tokenizer=text_tokenizer, embedding_model=embedding_model, model=model, device=device
    )

    sample_data = sample_data.to(device)
    # tokenized_caption = out_target_tokens.to(device)
    generated_captions = []

    if cfg.display.initial_caption:
        initial_caption = caption_preder.make_caption(
            image=sample_data,
            max_length=_max_length,
        )
        generated_captions.append(initial_caption)

    collate_fn = collate_func_helper(device=device, return_tensors="pt")

    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    train_dataloader = DataLoader(summary_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    display = GeneralistDisplay.make(display=display_flag, logger=log)
    display.manage()
    for epoch in range(n_epochs):
        # epoch_progress.update(epoch_task)

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        display.update("epoch_progress", epoch)
        display.add_task("batch_progress", "[green]Batch", total=len(train_dataloader), running_loss=running_loss)

        model.train()

        for batch_idx, batch in enumerate(train_dataloader):

            data, target = batch.data, batch.target

            target_mask = batch.get_masks("target")
            data_mask = batch.get_masks("data")

            # breakpoint()
            # for modality, data in batch.data.items():
            #     embedding[modality] = embedding_model(data)

            embedding = torch.cat([embedding_model(v) for modality, v in data.items()])

            # embedding = embedding_model(data)

            target_mask = ~target_mask.to(bool)

            # embedded_tgt = embedding_model.embed_data(task_input_ids, data_type="text")

            # emb = [embedding_model.embed_data(t_, data_type="text") for t_ in target]
            embedded_tgt = embedding_model(target)

            tgt_mask = model.get_tgt_mask(embedded_tgt=embedded_tgt)

            logits = model(embedding, embedded_tgt=embedded_tgt, tgt_key_padding_mask=target_mask, tgt_mask=tgt_mask)

            loss = loss_fn(logits[:, :-1].permute(0, 2, 1), target[:, 1:])

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            try:
                test_decoded = text_tokenizer.batch_decode(logits.argmax(dim=-1))
                # test_actual = text_tokenizer.batch_decode(encoded_target)
                test_actual = [t.data for t in target]

            except IndexError:
                breakpoint()

            batch_correct = sum([1 if a == b else 0 for a, b in zip(test_decoded, test_actual)])
            batch_total = len(test_decoded)

            running_correct += batch_correct
            running_total += batch_total

            # breakpoint()
            if batch_idx % 50 == 0:
                decoded__ = text_tokenizer.batch_decode(logits.argmax(dim=-1)[:5, :10])
                actual__ = text_tokenizer.batch_decode(target[:5, :10])
                print(list(zip(decoded__, actual__)))

                # breakpoint()

            if batch_idx % 250 == 0:
                latest_caption = caption_preder.make_caption(
                    image=sample_data,
                    max_length=context_length,
                )

                print(f"latest caption:{text_tokenizer.decode(latest_caption)}")
                print(f"true caption: {out_target_true}")

                generated_captions.append(latest_caption)

            acc = f"{(running_correct / running_total):0.3f}"

            display_vals = {
                "acc": acc,
                "batch_acc": batch_correct / batch_total,
                "batch_idx": batch_idx,
            }

            display.update(
                "batch_progress",
                advance=1,
                batch_loss=f"{loss.item():0.3f}",
                running_loss=f"{running_loss:.3f}",
                test=display_vals,
            )

        # save_checkpoint(model_save_dir, model, embedding_model, optimizer, epoch)
        save_checkpoint(
            model_save_dir,
            obj={
                "model": model,
                "embedding_model": embedding_model,
                "optimizer": optimizer,
                "epoch": epoch,
                "tokenizers": {
                    "image_tokenizer": image_tokenizer,
                    "text_tokenizer": text_tokenizer,
                },
            },
            filename="latest.pt",
        )

        latest_caption = caption_preder.make_caption(image=sample_data, max_length=context_length)
        generated_captions.append(latest_caption)

    captions_generated = text_tokenizer.batch_decode(generated_captions)
    print("--- --- --- captions_generated --- --- ---")
    print(captions_generated)

    display.manage("epoch", display.END)
    print("done with training")


if __name__ == "__main__":
    train()
