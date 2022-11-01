from enum import Enum
import logging
from typing import Any, Callable, Dict, Iterable, List

import wandb
import omegaconf
from rich import print as rich_print
from rich.console import Console


class DisplayStates(Enum):
    INIT = "init"
    RUN = "run"
    END = "end"


class IterHelper:
    def __init__(self, iterable: Iterable | int = None):
        self.task(iterable)

    def __iter__(self):
        for i in self._iterable:
            yield i

    def task(self, iterable: Iterable | int):
        if iterable:
            self._iterable = range(iterable) if isinstance(iterable, int) else iterable


class GeneralistDisplay(object):
    """
    base class for displaying info.
    just print info to console unless display
    """

    def __init__(self, **kwargs):
        # cant remember what these are for
        self.live = None
        self.tasks = {}

        self.wandb = None

    # def __getattr__(self, name: str) -> Any:
    #     if name == "epoch_progress" and not hasattr(self, name):
    #         setattr(self, name, IterHelper)

    @classmethod
    def make(cls, *args, **kwargs):
        cls = GeneralistDisplayLogger

        if kwargs.get("display", True):
            from generalist.utils.display.rich_display import RichDisplay

            cls = RichDisplay

        return cls(*args, **kwargs)

    def wandb_log(self, *args, **kwargs):
        pass

    def manage(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass

    def extra_setup(self, *args, **kwargs):
        pass

    def setup_layout(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def add_task(self, *args, **kwargs):
        pass


class GeneralistDisplayLogger(GeneralistDisplay):
    """display info with logger"""

    def __init__(self, display: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.console = Console()

        self._display = display
        if self._display is False:
            self.setup_iters(**kwargs)

        self.wandb = None
        cfg = kwargs.get("cfg", None)
        cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        if wandb_info := kwargs.get("wandb_info", None):

            if wandb_info.enabled:

                self.wandb = wandb
                self.wandb.config = cfg

                wandb_init_kwargs = {
                    "project": wandb_info.project,
                }

                if wandb_info.tags:
                    wandb_init_kwargs["tags"] = wandb_info.tags

                self.wandb.init(**wandb_init_kwargs)

    def extra_setup(self, model=None):
        if model and self.wandb:
            # self.wandb.watch(model)
            return

    def setup_iters(self, *args, **kwargs):
        self.logger = kwargs.get("logger", rich_print)

        self.epoch_progress = IterHelper()
        self.batch_progress = IterHelper()

        setattr(self, "update", self.setup_update(update_type="print"))

        if isinstance(self.logger, logging.Logger) and (not kwargs.get("override_rich_print")):
            setattr(self, "update", self.setup_update(update_type="log"))

    def setup_update(self, update_type: str = "print"):
        def update_log(*args, **kwargs):
            self.logger.info(f"{args} {kwargs}")

        def update_print(*args, **kwargs):
            rich_print(f"{args}, {kwargs}")

        update_fn = update_log if update_type == "log" else update_print

        def _update(*args, **kwargs):
            if self.wandb:
                if isinstance(args[0], dict):
                    self.wandb.log(args[0])
                if len(kwargs) > 0:
                    self.wandb.log(kwargs)

            update_fn(*args, **kwargs)

        return _update
