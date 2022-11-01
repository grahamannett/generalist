from dataclasses import dataclass
from typing import List
from omegaconf import DictConfig
from generalist.train import train_step
import hydra
from hydra import compose, initialize
from generalist.utils.utils import combine_cfgs, get_hostname


@dataclass
class ExperimentHelper:
    model = None

    def __post_init__(self):
        pass


class ResultsTracker:
    def __init__(self, wandb):
        self.wandb = wandb


def get_results():
    pass


def exp_combine_tgt():
    with initialize(version_base=None, config_path="../../config"):
        base_cfg = compose(config_name=get_hostname())

    with initialize(config_path="."):
        exp_cfg = hydra.compose(config_name="experiment")

    results_tracker = ResultsTracker()

    cfg = combine_cfgs(base_cfg, exp_cfg)

    experiment_related = ExperimentHelper(cfg)

    for epoch in range(cfg.training.n_epochs):
        train_step(experiment_related, cfg, results_tracker)
        results = get_results()

    # train_step(c)


if __name__ == "__main__":
    exp_combine_tgt()
