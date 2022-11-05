from dataclasses import dataclass, make_dataclass
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich import print
from torch.utils.data import DataLoader

from generalist.generalist_datasets.coco.coco import (
    CocoCaption,
    CocoCaptionTargetTransform,
    CocoFilepaths,
    CocoImageTransforms,
)
from generalist.eval import eval_summary, preliminary_eval
from generalist.generalist_datasets.dataset_utils import DatasetRegistry
from generalist.generalist_datasets.hf.summary import BillSum, XSum, SummaryTransforms
from generalist.generalist_datasets.utils.data_collate import collate_func_helper
from generalist.generalist_datasets.utils.multiple_datasets import BatchUniformDatasetSampler, CombinedDataset
from generalist.generalist_tokenizers import image_tokenizers, text_tokenizers
from generalist.models.model import GeneralistModel
from generalist.models.output_model import GeneralOutput
from generalist.utils.display.display import GeneralistDisplay
from generalist.utils.utils import get_hostname, save_checkpoint


# from rich.progress import Progress, track
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class StepStats:
    loss: float
    scores: Any = None

    def values(self):
        _vals = {"loss": self.loss}
        if self.scores is not None:
            _vals["scores"] = self.scores
        return _vals


@dataclass
class Tokenizers:
    image: image_tokenizers.ImageTokenizer
    text: text_tokenizers.TextTokenizer
    text_tokenizer_encode_kwargs: DictConfig


def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    use_progress_bar: bool = True,
    end_early: int = None,
):
    if use_progress_bar:
        dataloader = tqdm(dataloader)

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

        if end_early and end_early == batch_idx:
            break

    if scheduler:
        scheduler.step()

    return StepStats(loss=running_loss)


def test_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    use_progress_bar: bool = True,
    text_tokenizer: text_tokenizers.TextTokenizer = None,
    end_early: int = None,
):
    if use_progress_bar:
        dataloader = tqdm(dataloader)
    #     dataloader = track(dataloader, description="[yellow]Test Step [blue]Batch...", disable=True)

    all_predictions = []
    all_references = []

    model.eval()
    running_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        data, targets = batch.data, batch.target

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

        if text_tokenizer:
            batch_predictions = text_tokenizer.batch_decode(logits.argmax(dim=-1)[:5], skip_special_tokens=True)
            batch_references = text_tokenizer.batch_decode(targets[:5], skip_special_tokens=True)
            all_predictions.extend(batch_predictions)
            all_references.extend(batch_references)

        if end_early and end_early == batch_idx:
            break

    stats = StepStats(loss=running_loss)

    if len(all_predictions) > 0:
        stats.scores = eval_summary(predictions=all_predictions, references=all_references)

    return stats


def get_tokenizers(cfg: DictConfig) -> Tokenizers:
    image_tokenizer = image_tokenizers.ImageTokenizer(device=cfg.device)
    text_tokenizer = text_tokenizers.TextTokenizerBert.from_pretrained(cfg.text_tokenizer.name)

    tokenizers = Tokenizers(image=image_tokenizer, text=text_tokenizer, text_tokenizer_encode_kwargs=cfg.text_tokenizer.encode_kwargs)

    return tokenizers


def _get_datasets_split(cfg: DictConfig, split_name, tokenizers):
    datasets = []
    datasets_info = cfg.datasets[split_name]
    for dataset_name, dataset_info in datasets_info.items():
        split = dataset_info.get("split", split_name)

        dataset_cls = DatasetRegistry[dataset_name]
        dataset = dataset_cls.from_cfg(split=split, cfg=cfg, tokenizers=tokenizers)
        datasets.append(dataset)
    return datasets


def get_datasets_from_cfg(cfg: DictConfig, tokenizers: Tokenizers) -> Dict[str, DataLoader]:

    train_datasets = _get_datasets_split(cfg=cfg, split_name="train", tokenizers=tokenizers)
    test_dataset = _get_datasets_split(cfg=cfg, split_name="test", tokenizers=tokenizers)
    return CombinedDataset(train_datasets), CombinedDataset(test_dataset)


def get_datasets(cfg: DictConfig, tokenizers: Tokenizers):
    device = cfg.device
    batch_size = cfg.training.batch_size
    coco_filepaths = CocoFilepaths(base_dir=cfg.coco_dir, split="train")
    coco_filepaths_test = CocoFilepaths(base_dir=cfg.coco_dir, split="val")

    coco_target_transform = CocoCaptionTargetTransform.get(
        text_tokenizer=tokenizers.text, text_tokenizer_kwargs=tokenizers.text_tokenizer_encode_kwargs
    )

    coco_caption = CocoCaption.from_base_dir(
        base_dir=cfg.coco_dir,
        split="train",
        text_tokenizer=tokenizers.text,
        text_tokenizer_kwargs=tokenizers.text_tokenizer_encode_kwargs,
        transform=CocoImageTransforms.train,
        target_transform=coco_target_transform.train,
    )

    summary_transforms = SummaryTransforms.make_transforms(
        text_tokenizer=tokenizers.text, text_tokenizer_kwargs=tokenizers.text_tokenizer_encode_kwargs
    )

    summary_dataset = XSum(
        split="train",
        text_transform=summary_transforms.train,
    )

    chained_dataset = CombinedDataset([coco_caption, summary_dataset])

    coco_caption_test = CocoCaption(
        root=coco_filepaths_test.images_root,
        annFile=coco_filepaths_test.captions_filepath,
        transform=CocoImageTransforms.train,
        target_transform=coco_target_transform.test,
    )

    summary_dataset_test = XSum(
        text_transform=summary_transforms.test,
        split="test",
    )
    chained_dataset_test = CombinedDataset([coco_caption_test, summary_dataset_test])

    return chained_dataset, chained_dataset_test


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

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = count_parameters(model)
    # breakpoint()

    display.extra_setup(model=model)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.embedding_model.parameters()},
            {"params": model.transformer_encoder.parameters()},
            {"params": model.transformer_decoder.parameters()},
            {"params": model.output_model.parameters()},
        ],
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    text_tokenizer_encode_kwargs = cfg.text_tokenizer.encode_kwargs

    train_dataset, test_dataset = get_datasets(
        cfg, image_tokenizer=image_tokenizer, text_tokenizer=text_tokenizer, text_tokenizer_encode_kwargs=text_tokenizer_encode_kwargs
    )

    collate_fn = collate_func_helper(device=device, return_tensors="pt")
    if cfg.training.batch_uniform_dataset_samples:
        batch_sampler = BatchUniformDatasetSampler(datasets=train_dataset, batch_size=batch_size)
        train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    else:
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    summary_dataset = train_dataset.datasets[1]
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

    display.setup_layout(debug=False)

    display.epoch_progress.task(n_epochs)

    save_data = {
        "train": [],
        "test": [],
    }

    for epoch in display.epoch_progress:

        display.batch_progress.task(train_dataloader)

        train_step_stats = train_step(model=model, dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn, scheduler=scheduler)
        test_step_stats = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, text_tokenizer=text_tokenizer)

        display.wandb.log({"train_loss": train_step_stats.loss})

        display.update(
            "epoch_done", epoch, train_loss=train_step_stats.loss, test_loss=test_step_stats.loss, test_scores=test_step_stats.scores
        )

        save_data["train"].append(train_step_stats)

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

    torch.save(save_data, "experiments/batch_composition/save_data.pt")
    print("done with training")


if __name__ == "__main__":
    train()
