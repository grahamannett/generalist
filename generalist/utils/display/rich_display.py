import atexit

from typing import Any, Dict, List
from rich import box, print
from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from generalist.utils.display.display import GeneralistDisplay


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


class BatchProgress:
    def __init__(self, n_batches: int):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            SpinnerColumn("simpleDots"),
            BarColumn(bar_width=None),
        )

        self.progress.add_task("[bold blue]Batch", total=n_batches)
        self._panel = Panel(self.progress)


class RichDisplay(GeneralistDisplay):
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


if __name__ == "__main__":
    import time

    def get_args():
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true")
        return parser.parse_args()

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
    args = get_args()
    # breakpoint()

    # args.
    debug = args.debug

    display = RichDisplay()
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
