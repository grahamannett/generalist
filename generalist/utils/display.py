from typing import Callable
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


class GeneralistDisplay:
    def __init__(self, live: Live, display: bool = True):
        self.live = live

        self._status = "init" if display else False

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value

    def manage(self):

        match self.status:
            case "init":
                self.status = "running"
                return self.live.start(True)
            case "running":
                self.status = "ended"
                return self.live.stop()
            case False:
                return None
            case _:
                return None

    def setup(self, n_epochs: int = None):
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

        progress_group = Group(
            Panel(self.epoch_progress),
            Panel(Group(self.batch_progress)),
        )

        # epoch_task = epoch_progress.add_task("[bold blue]Epoch", total=n_epochs)
        epoch_task = self.epoch_progress.add_task("[bold blue]Epoch")
        self.live = Live(progress_group)

    def update(self, epoch_kwargs: dict = None, batch_kwargs: dict = None):
        if epoch_kwargs:
            self.epoch_progress.update(**epoch_kwargs)

    def add_task(self, name: str, obj: Any, **kwargs):

        # self.batch_task = self.batch_progress.add_task(
        #     "[green]Batch", total=len(train_dataloader), running_loss=running_loss
        # )
        obj = obj.add_task(**kwargs)
        setattr(self, name, obj)
