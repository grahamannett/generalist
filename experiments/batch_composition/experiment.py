from pathlib import Path
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from generalist.generalist_datasets.dataset_utils import DatasetRegistry

from generalist.generalist_datasets.utils.data_collate import collate_func_helper
from generalist.generalist_datasets.utils.multiple_datasets import BatchUniformDatasetSampler
from generalist.generalist_tokenizers import image_tokenizers, text_tokenizers

from generalist.models.model import GeneralistModel
from generalist.train import get_datasets, get_tokenizers, test_step, train_step, get_datasets_from_cfg
from generalist.utils.utils import combine_cfgs, get_hostname

import wandb


def main():
    with hydra.initialize(version_base=None, config_path="../../config"):
        base_cfg = hydra.compose(config_name=get_hostname())

    with hydra.initialize(config_path="."):
        exp_cfg = hydra.compose(config_name="exp")

    cfg = combine_cfgs(base_cfg, exp_cfg)

    wandb.init(**cfg.wandb)

    model_save_dir = Path(cfg.model_save_dir)
    device = cfg.device
    context_length = cfg.context_length

    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    n_epochs = cfg.training.n_epochs

    model_dim = cfg.model.model_dim

    tokenizers = get_tokenizers(cfg=cfg)
    train_dataset, test_dataset = get_datasets_from_cfg(cfg=cfg, tokenizers=tokenizers)

    model = GeneralistModel(**cfg.model).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
            }
        ],
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    train_dataset, test_dataset = get_datasets(cfg, tokenizers=tokenizers)

    collate_fn = collate_func_helper(device=device, return_tensors="pt")
    if cfg.training.batch_uniform_dataset_samples:
        batch_sampler = BatchUniformDatasetSampler(datasets=train_dataset, batch_size=batch_size)
        train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    else:
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    loss_fn = torch.nn.CrossEntropyLoss()

    step_stats = {"train": [], "test": []}

    for epoch in range(n_epochs):
        train_step_stats = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            end_early=cfg.training.end_early,
        )
        test_step_stats = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, text_tokenizer=tokenizers.text, end_early=cfg.training.end_early
        )

        wandb.log({"train_loss": train_step_stats.loss, "test_loss": test_step_stats.loss, "scores": test_step_stats.scores}, step=epoch)

        step_stats["train"].append(train_step_stats)
        step_stats["test"].append(test_step_stats)

    if cfg.stats_file:
        torch.save(step_stats, cfg.stats_file)


if __name__ == "__main__":
    main()
