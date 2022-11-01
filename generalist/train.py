from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

import hydra
import torch
from omegaconf import DictConfig
from rich import print
from torch.utils.data import DataLoader

from generalist.generalist_datasets.coco.coco import (
    CocoCaption,
    CocoCaptionTargetTranform,
    CocoFilepaths,
    CocoImageTransforms,
)
from generalist.eval import preliminary_eval
from generalist.generalist_datasets.hf.summary import BillSum, XSum, SummaryTransforms
from generalist.generalist_datasets.utils.data_collate import collate_func_helper
from generalist.generalist_datasets.utils.multiple_datasets import BatchUniformDatasetSampler, CombinedDataset
from generalist.generalist_tokenizers import image_tokenizers, text_tokenizers
from generalist.models.embedding_model import EmbeddingModel
from generalist.models.model import GeneralistModel
from generalist.models.output_model import GeneralOutput
from generalist.predict import ImageCaptionPrediction
from generalist.utils.display.display import GeneralistDisplay
from generalist.utils.utils import get_hostname, save_checkpoint

from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.bleu import bleu_score

from rich.progress import Progress, track

log = logging.getLogger(__name__)

@dataclass
class StepStats:
    loss: float


def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    use_progress_bar: bool = True,
    text_tokenizer: text_tokenizers.TextTokenizer = None,
):
    if use_progress_bar:
        dataloader = track(dataloader, description="[green]Train Step [blue]Batch...")

    model.train()
    running_loss = 0
    for batch_idx, batch in enumerate(dataloader):

        data, targets = batch.data, batch.target
        task_types = batch.tasks
        if len(set(task_types)) > 1:
            raise ValueError("Batch contains multiple task types!!!")

        target_mask = batch.get_masks("target")
        target_mask = ~target_mask.to(bool)

        embedded_data = model.embedding(data)
        embedded_tgt = model.embedding(targets)

        tgt_mask = model.get_tgt_mask_tri(embedded_tgt=embedded_tgt)

        if model.combine_embeddings:
            logits = model(embedded_src=torch.cat([embedded_data, embedded_tgt], dim=1))
            logits = logits[:, embedded_tgt.shape[1] :]
        else:
            logits = model(embedded_src=embedded_data, embedded_tgt=embedded_tgt, tgt_key_padding_mask=target_mask, tgt_mask=tgt_mask)

        loss = loss_fn(logits[:, :-1].permute(0, 2, 1), targets[:, 1:])

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return StepStats(loss=running_loss)


def test_step(model: torch.nn.Module, dataloader: DataLoader, loss_fn: Callable, use_progress_bar: bool = True):
    if use_progress_bar:
        dataloader = track(dataloader, description="[yellow]Test Step [blue]Batch...")

    model.eval()
    running_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        data, targets = batch.data, batch.target
        task_types = batch.tasks

        target_mask = batch.get_masks("target")
        target_mask = ~target_mask.to(bool)

        embedded_data = model.embedding(data)
        embedded_tgt = model.embedding(targets)

        tgt_mask = model.get_tgt_mask_tri(embedded_tgt=embedded_tgt)

        if model.combine_embeddings:
            logits = model(embedded_src=torch.cat([embedded_data, embedded_tgt], dim=1))
            logits = logits[:, embedded_tgt.shape[1] :]
        else:
            logits = model(embedded_src=embedded_data, embedded_tgt=embedded_tgt, tgt_key_padding_mask=target_mask, tgt_mask=tgt_mask)

        loss = loss_fn(logits[:, :-1].permute(0, 2, 1), targets[:, 1:])

        running_loss += loss.item()

    return StepStats(loss=running_loss)


def post_batch_callback(logits, targets, text_tokenizer, batch_idx, display, running_loss, loss):
    test_decoded = text_tokenizer.batch_decode(logits.argmax(dim=-1))
    test_actual = [t.data for t in targets]

    batch_total = len(test_decoded)

    running_correct += batch_correct
    running_total += batch_total

    # breakpoint()
    if batch_idx % 50 == 0:
        batch_predictions = text_tokenizer.batch_decode(logits.argmax(dim=-1)[:5], skip_special_tokens=True)
        batch_references = text_tokenizer.batch_decode(targets[:5], skip_special_tokens=True)

        scores = eval_summary(predictions=batch_predictions, references=batch_references)
        display.update(scores)
        print(list(zip(batch_predictions, batch_references)))

    acc = f"{(running_correct / running_total):0.3f}"

    display_vals = {
        "acc": acc,
        "batch_acc": batch_correct / batch_total,
        "batch_idx": batch_idx,
    }

    display.update(
        "batch_progress",
        batch_loss=f"{loss.item():0.3f}",
        running_loss=f"{running_loss:.3f}",
        test=display_vals,
    )


@hydra.main(config_path=f"../config", config_name=get_hostname(), version_base=None)
def train(cfg: DictConfig):
    display = GeneralistDisplay.make(display=cfg.display.display_flag, logger=log, override_rich_print=True, wandb_info=cfg.wandb, cfg=cfg)

    model_save_dir = Path(cfg.model_save_dir)
    device = cfg.device
    context_length = cfg.context_length

    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    n_epochs = cfg.training.n_epochs

    model_dim = cfg.model.model_dim

    image_tokenizer = image_tokenizers.ImageTokenizer(device=device)
    text_tokenizer = text_tokenizers.TextTokenizerBert.from_pretrained("bert-base-uncased")

    output_model = GeneralOutput(model_dim=model_dim, output_dim=text_tokenizer.vocab_size)
    model = GeneralistModel(output_model=output_model, **cfg.model).to(device)
    embedding_model = model.embedding_model

    model.to(device)

    display.extra_setup(model=model)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            # {"params": embedding_model.parameters()},
            {"params": model.embedding_model.parameters()},
            {"params": model.transformer_encoder.parameters()},
            {"params": model.transformer_decoder.parameters()},
            {"params": model.output_model.parameters()},
        ],
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    tokenizers = [image_tokenizer, text_tokenizer]

    text_tokenizer_encode_kwargs = cfg.text_tokenizer.encode_kwargs

    coco_filepaths = CocoFilepaths(base_dir=cfg.coco_dir, split="train")

    coco_caption = CocoCaption(
        root=coco_filepaths.images_root,
        annFile=coco_filepaths.captions_filepath,
        transform=CocoImageTransforms.train,
        target_transform=CocoCaptionTargetTranform.get(
            text_tokenizer=text_tokenizer, text_tokenizer_kwargs=text_tokenizer_encode_kwargs
        ).train,
    )
    summary_transforms = SummaryTransforms.make_transforms(
        text_tokenizer=text_tokenizer, text_tokenizer_kwargs=text_tokenizer_encode_kwargs
    )

    summary_dataset = XSum(
        split="train",
        text_transform=summary_transforms.train,
    )

    # summary_dataset_billsum = BillSum(
    #     text_transform=SummaryTransforms.make_transforms(text_tokenizer=text_tokenizer, text_tokenizer_kwargs=text_tokenizer_kwargs).train,
    # )

    dataset = coco_caption
    chained_dataset = CombinedDataset([coco_caption, summary_dataset])

    concat_datasets = torch.utils.data.ConcatDataset([coco_caption, summary_dataset])

    coco_filepaths_test = CocoFilepaths(base_dir=cfg.coco_dir, split="val")
    coco_target_tranfroms = CocoCaptionTargetTranform.get(text_tokenizer=text_tokenizer, text_tokenizer_kwargs=text_tokenizer_encode_kwargs)
    coco_caption_test = CocoCaption(
        root=coco_filepaths_test.images_root,
        annFile=coco_filepaths_test.captions_filepath,
        transform=CocoImageTransforms.train,
        target_transform=coco_target_tranfroms.test,
    )
    summary_dataset_test = XSum(
        text_transform=summary_transforms.test,
        split="test",
    )
    chained_dataset_test = CombinedDataset([coco_caption_test, summary_dataset_test])

    sample = summary_dataset[0]
    sample.data = sample.data.to(device)
    sample.target = sample.target.to(device)

    generated_captions, caption_preder, out_target_true = preliminary_eval(
        sample,
        text_tokenizer=text_tokenizer,
        image_tokenizer=image_tokenizer,
        model=model,
        device=device,
        initial_caption=cfg.predictions.initial_generation,
    )

    collate_fn = collate_func_helper(device=device, return_tensors="pt")

    if cfg.training.batch_uniform_dataset_samples:
        sampler = BatchUniformDatasetSampler(datasets=chained_dataset, batch_size=batch_size)
        train_dataloader = DataLoader(dataset=chained_dataset, batch_sampler=sampler, collate_fn=collate_fn)
    else:
        train_dataloader = DataLoader(chained_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    display.setup_layout(debug=False)

    display.epoch_progress.task(n_epochs)

    for epoch in display.epoch_progress:

        display.batch_progress.task(train_dataloader)

        train_step_stats = train_step(
            model=model, dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn, text_tokenizer=text_tokenizer
        )
        test_step_stats = test_step(model=model, dataloader=chained_dataset_test, loss_fn=loss_fn)

        display.wandb.log({"train_loss": train_step_stats.loss})

        display.update("epoch_done", epoch, train_loss=train_step_stats.loss, test_loss=test_step_stats.loss)

        # save_checkpoint(model_save_dir, model, embedding_model, optimizer, epoch)
        save_checkpoint(
            model_save_dir,
            obj={
                "model": model,
                "optimizer": optimizer,
                "epoch": epoch,
                "tokenizers": {
                    "image_tokenizer": image_tokenizer,
                    "text_tokenizer": text_tokenizer,
                },
            },
            filename="latest.pt",
        )

    print("done with training")


if __name__ == "__main__":
    train()
