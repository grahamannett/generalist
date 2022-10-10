from typing import Any
from omegaconf import DictConfig
from hydra.experimental.callback import Callback


class SetupRuns(Callback):
    def __init__(self) -> None:
        pass

    def on_run_start(self, cfg: DictConfig, **kwargs: Any) -> None:
        print("callback with run start")

    # def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
    #     print(f"Job started, downloading...")
    #     breakpoint()
