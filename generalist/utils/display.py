from typing import Any, Callable

import atexit

from rich import print
from rich.live import Live
from rich.console import Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    SpinnerColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
)
from rich.text import Text
from rich.layout import Layout


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

    def __init__(self, display: bool, **kwargs):
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

    def _stop(self):
        self.live.stop()

    def setup_layout(self):
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=1),
        )
        self.layout["header"].update(Panel("Header"))
        self.layout["body"].update(Panel("Body"))
        self.layout["footer"].update(Panel("Footer"))

        self.live = Live(self.layout, refresh_per_second=10)
        self.live.start()

    def setup(self, n_epochs: int = None):
        self._status = "run"

        self.layout = Layout(name="root")

        self.epoch_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            SpinnerColumn("simpleDots"),
            BarColumn(bar_width=None),
        )
        epoch_panel = Panel(self.epoch_progress)

        self.batch_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            TextColumn(
                "[progress.percentage]{task.percentage:>3.0f}% \t [bold purple]Loss=>{task.fields[running_loss]:.3f}%[reset]",
            ),
            transient=True,
        )
        # batch_text =
        batch_panel = Panel(
            self.batch_progress,
        )

        # self.text_group = generate_table()

        self.progress_group = Group(epoch_panel)

        self.epoch_task = self.epoch_progress.add_task("[bold blue]Epoch", total=n_epochs)
        self.live = Live(self.progress_group)
        self.live.start(True)

        atexit.register(self._stop)
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
        # print("stopping!!")
        # self.live.stop()
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

        # self.live.update(generate_table())

    def add_task(self, name: str, *args, **kwargs):
        obj = getattr(self, name)

        self.tasks[name] = obj.add_task(**kwargs)

        # self.batch_task = self.batch_progress.add_task(
        #     "[green]Batch", total=len(train_dataloader), running_loss=running_loss
        # )
        # obj = obj.add_task(**kwargs)
        setattr(self, name, obj)


if __name__ == "__main__":
    import time

    display = GeneralistDisplayRich()
    num_epochs = 3
    display.setup(n_epochs=num_epochs)

    for i in range(num_epochs):
        display.update(epoch_kwargs={"task_id": display.epoch_task, "advance": 1})
        time.sleep(1)

    # display.stop()
