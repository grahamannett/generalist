import atexit
from pickle import OBJ
from typing import Any, Callable, Dict, List

from rich import box, print
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def make_sponsor_message(new_info=None) -> Panel:
    """Some example content."""
    sponsor_message = Table.grid(padding=1)
    sponsor_message.add_column(header="abd", style="green", justify="right")
    sponsor_message.add_column(header="dfg")
    sponsor_message.add_row(
        "Twitter",
        "[u blue link=https://twitter.com/textualize]https://twitter.com/textualize",
        end_section=True,
    )

    if new_info:
        sponsor_message.add_row(new_info[0], new_info[1], end_section=True)
    else:
        sponsor_message.add_row(
            "CEO",
            "[u blue link=https://twitter.com/willmcgugan]https://twitter.com/willmcgugan",
        )
    sponsor_message.add_row("Textualize", "[u blue link=https://www.textualize.io]https://www.textualize.io")

    # message = Table.grid(padding=5)
    # message.add_column()
    # # message.add_column(no_wrap=True)
    # message.add_row(sponsor_message)

    message_panel = Panel(
        Group(sponsor_message),
        # Align.center(
        #     Group("\n", Align.center(sponsor_message)),
        #     vertical="middle",
        # ),
        box=box.ROUNDED,
        padding=(1, 2),
        title="[b red]Thanks for trying out Rich!",
        border_style="bright_blue",
    )
    return message_panel


class SampleInfo:
    def __init__(self, parent_layout: Layout, title: str = None, padding: int = 1):
        self.parent_layout = parent_layout
        self.padding = padding
        self.title = title

    def make_table(self, data: List[Dict[str, Any]]):
        for col in list(data[0].keys()):
            self.table.add_column(col)

        for row in data:
            self.table.add_row(*[f"{val}" for val in row.values()], end_section=True)

    def update(self, data: List[Dict[str, Any]] = None):

        self.table = Table.grid(padding=self.padding)

        if data:
            self.make_table(data)

        panel = Panel(
            Group(self.table),
            box=box.ROUNDED,
            padding=(1, 2),
            title=self.title,
            border_style="bright_blue",
        )

        self.parent_layout.update(panel)


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


class EpochProgress:
    def __init__(self, n_epochs: int):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            SpinnerColumn("simpleDots"),
            BarColumn(bar_width=None),
        )

        self.progress.add_task("[bold blue]Epoch", total=n_epochs)
        self._panel = Panel(self.progress)

    @property
    def panel(self):
        return self._panel

    def advance(self):
        self.progress.advance(0)


class GeneralistDisplayRich(GeneralistDisplay):
    def __init__(self, display: bool = True):
        super().__init__(display)
        self._status = "init" if display else False

    def _stop(self):
        self.live.stop()

    def run(self, debug: bool = False):
        if not debug:
            self.live.start()
        else:
            self.debug_run()

    def debug_run(self):
        self.sample_info.update = print


    def setup_layout(self, n_epochs: int, **kwargs):
        self.layout = Layout(name="root")

        self.layout.split(
            Layout(name="main", ratio=2),
            Layout(name="sample_info"),
        )

        # self.layout["main"].update(self.setup_epoch_progress(n_epochs=n_epochs))
        self.epoch_progress = EpochProgress(n_epochs=n_epochs)

        self.sample_info = SampleInfo(parent_layout=self.layout["sample_info"], title="Sample Info", padding=1)

        self.layout["main"].update(self.epoch_progress.panel)

        self.live = Live(self.layout, refresh_per_second=10, screen=True)
        atexit.register(self._stop)

    def update_sample_info(self, new_info):
        self.sample_info.update(new_info)

    def advance_epoch(self):
        self.epoch_progress.advance()
        # self.epoch_progress.advance()

    # def setup(self, n_epochs: int = None):
    #     self._status = "run"

    #     self.layout = Layout(name="root")

    #     self.epoch_progress = Progress(
    #         TextColumn("[progress.description]{task.description}"),
    #         MofNCompleteColumn(),
    #         SpinnerColumn("simpleDots"),
    #         BarColumn(bar_width=None),
    #     )
    #     epoch_panel = Panel(self.epoch_progress)

    #     self.batch_progress = Progress(
    #         SpinnerColumn(),
    #         TextColumn("[progress.description]{task.description}"),
    #         MofNCompleteColumn(),
    #         TextColumn(
    #             "[progress.percentage]{task.percentage:>3.0f}% \t [bold purple]Loss=>{task.fields[running_loss]:.3f}%[reset]",
    #         ),
    #         transient=True,
    #     )
    #     # batch_text =
    #     batch_panel = Panel(
    #         self.batch_progress,
    #     )

    #     # self.text_group = generate_table()

    #     self.progress_group = Group(epoch_panel)

    #     self.epoch_task = self.epoch_progress.add_task("[bold blue]Epoch", total=n_epochs)
    #     self.live = Live(self.progress_group)
    #     self.live.start(True)

    #     atexit.register(self._stop)
    #     return self

    # def manage(self):
    #     match self._status:
    #         case False:
    #             pass
    #         case "init":
    #             self.setup()
    #         case "run":
    #             self.stop()
    #         case _:
    #             pass

    #     return self

    # def track(self, *args, **kwargs):
    #     print(**kwargs)

    # def stop(self):
    #     # print("stopping!!")
    #     # self.live.stop()
    #     self._status = "ended"
    #     return self

    # def update(self, epoch_kwargs: dict = None, batch_kwargs: dict = None):
    #     # batch_progress.update(
    #     #     batch_task,
    #     #     advance=1,
    #     #     running_loss=running_loss,
    #     # )
    #     if epoch_kwargs:
    #         self.epoch_progress.update(**epoch_kwargs)

    #     # self.live.update(generate_table())

    # def add_task(self, name: str, *args, **kwargs):
    #     obj = getattr(self, name)

    #     self.tasks[name] = obj.add_task(**kwargs)

    #     # self.batch_task = self.batch_progress.add_task(
    #     #     "[green]Batch", total=len(train_dataloader), running_loss=running_loss
    #     # )
    #     # obj = obj.add_task(**kwargs)
    #     setattr(self, name, obj)


sample_info_arr = [
    [{"idx": 1, "text": "This is a text1"}, {"idx": 1, "text": "This is a text1"}, {"idx": 1, "text": "This is a text1"}],
    [{"idx": 2, "text": "This is a text2"}, {"idx": 2, "text": "This is a text2"}],
    [
        {"idx": 3, "text": "This is a text3"},
        {"idx": 3, "text": "This is a text3"},
        {"idx": 3, "text": "This is a text3"},
        {"idx": 3, "text": "This is a text3"},
    ],
]

if __name__ == "__main__":
    import time

    args = get_args()
    # breakpoint()

    # args.
    debug = args.debug

    display = GeneralistDisplayRich()
    num_epochs = len(sample_info_arr)
    display.setup_layout(n_epochs=num_epochs)
    display.run(debug=args.debug)

    for i in range(num_epochs):
        # display.update(epoch_kwargs={"task_id": display.epoch_task, "advance": 1})
        time.sleep(1.5)
        display.advance_epoch()
        display.update_sample_info(sample_info_arr[i])
        time.sleep(1.5)

    display.stop()
    # display._stop()
