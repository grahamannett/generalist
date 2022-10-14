from enum import Enum
from typing import Any, Callable, Dict, List

from rich.console import Console


class DisplayStates(Enum):
    INIT = "init"
    RUN = "run"
    END = "end"


from rich import print


class GeneralistDisplay(object):
    """
    base class for displaying info.
    just print info to console unless display
    """

    def __init__(self, display: bool, **kwargs):
        self.console = Console()

        self._display = display
        self.live = None

        self.tasks = {}

    @classmethod
    def make(cls, *args, **kwargs):
        if kwargs.get("display", True):
            from generalist.utils.rich_display.rich_display import RichDisplay

            cls = RichDisplay
        return cls(*args, **kwargs)

    def setup(self, *args, **kwargs):
        pass

    def manage(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass

    def setup_layout(self, *args, **kwargs):
        pass

    def ready_batch(self, *args, **kwargs):
        pass

    def ready_epoch(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        # self.console.log(f"args={args}, kwargs={kwargs}")
        # self.console.log(f"{args} {kwargs}")
        print(f"{args} {kwargs}")

    def add_task(self, *args, **kwargs):
        pass
