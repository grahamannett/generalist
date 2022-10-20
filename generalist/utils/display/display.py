from enum import Enum
import logging
from typing import Any, Callable, Dict, Iterable, List

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

    # def __getattr__(self, name: str) -> Any:
    #     if name == "epoch_progress" and not hasattr(self, name):
    #         setattr(self, name, IterHelper)

    @classmethod
    def make(cls, *args, **kwargs):
        if kwargs.get("display", True):
            from generalist.utils.display.rich_display import RichDisplay

            cls = RichDisplay
        else:
            cls = GeneralistDisplayLogger

        return cls(*args, **kwargs)

    def manage(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass

    def setup_layout(self, *args, **kwargs):
        pass

    # def ready_batch(self, *args, **kwargs):
    #     pass

    # def ready_epoch(self, *args, **kwargs):
    #     pass

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

    def setup_iters(self, *args, **kwargs):
        self.logger = kwargs.get("logger", rich_print)

        self.epoch_progress = IterHelper()
        self.batch_progress = IterHelper()

        setattr(self, "update", self.update_print)

        if isinstance(self.logger, logging.Logger):
            setattr(self, "update", self.update_log)

        if kwargs.get("override_rich_print", False):
            setattr(self, "update", self.update_print)

    def update_log(self, *args, **kwargs):
        self.logger.info(f"{args} {kwargs}")

    def update_print(self, *args, **kwargs):
        rich_print(f"{args}, {kwargs}")
