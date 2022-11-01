#! /bin/bash

time HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online \
    python generalist/train.py \
    wandb.enabled=True \
    wandb.tags=[batch_uniform_dataset_samples] \
    training.n_epochs=20 \
    training.batch_uniform_dataset_samples=True \
    model.use_encoder=True \
    model.use_encoder=False \
    predictions.initial_generation=True \
    display.display_flag=False
