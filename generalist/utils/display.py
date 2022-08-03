from typing import Any, Callable

from rich import print
from rich.live import Live
from rich.console import Group
from rich.panel import Panel
from rich.live import Live
from rich.progress import (
    BarColumn,
    SpinnerColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
)

from rich.console import Console


def _dummy(*args, **kwargs):
    pass


class GeneralistDisplay(object):
    """
    base class for displaying info...
    just print info to console unless display
    """

    # enums on body
    INIT = "init"
    RUN = "run"
    END = "end"

    def __init__(self, display: bool):
        self.console = Console()

        self._display = display
        self.live = None

        self.tasks = {}

    @classmethod
    def make(cls, *args, **kwargs):
        if kwargs.get("display", True):
            cls = GeneralistDisplayRich
        return cls(*args, **kwargs)

    def setup(self, *args, **kwargs):
        pass

    def manage(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        # self.console.log(f"args={args}, kwargs={kwargs}")
        self.console.log(f"{args} {kwargs}")

    def add_task(self, *args, **kwargs):
        pass


class GeneralistDisplayRich(GeneralistDisplay):
    def __init__(self, display: bool = True):
        super().__init__(display)
        self._status = "init" if display else False

    def setup(self, n_epochs: int = None):
        self._status = "run"

        self.epoch_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            BarColumn(),
        )
        self.batch_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            TextColumn(
                "[progress.percentage]{task.percentage:>3.0f}% \t [bold purple]Loss=>{task.fields[running_loss]:.3f}%[reset]",
            ),
            transient=True,
        )

        self.progress_group = Group(
            Panel(self.epoch_progress),
            Panel(Group(self.batch_progress)),
        )

        self.epoch_task = self.epoch_progress.add_task("[bold blue]Epoch", total=n_epochs)
        self.live = Live(self.progress_group)
        self.live.start(True)
        return self

    def manage(self):
        match self._status:
            case False:
                pass
            case "init":
                self.setup()
            case "run":
                self.stop()
            case _:
                pass

        return self

    def track(self, *args, **kwargs):
        print(**kwargs)

    def stop(self):
        self.live.stop()
        self._status = "ended"
        return self

    def update(self, epoch_kwargs: dict = None, batch_kwargs: dict = None):
                # batch_progress.update(
            #     batch_task,
            #     advance=1,
            #     running_loss=running_loss,
            # )
        if epoch_kwargs:
            self.epoch_progress.update(**epoch_kwargs)

    def add_task(self, name: str, *args, **kwargs):
        obj = getattr(self, name)

        self.tasks[name] = obj.add_task(**kwargs)

        # self.batch_task = self.batch_progress.add_task(
        #     "[green]Batch", total=len(train_dataloader), running_loss=running_loss
        # )
        # obj = obj.add_task(**kwargs)
        setattr(self, name, obj)
