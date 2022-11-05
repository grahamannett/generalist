#! /bin/bash

time CUDA_VISIBLE_DEVICES=0 python experiments/batch_composition/experiment.py training.batch_uniform_dataset_samples=False training.n_epochs=25 wand.tags=["random_batches"] stats_file="random_sta
ts.pt"
