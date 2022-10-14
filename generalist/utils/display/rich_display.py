import atexit
from torch.utils.data import DataLoader

from typing import Any, Dict, Iterable, List
from rich import box, print
from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, ProgressColumn
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
        self._panel = panel

        self.parent_layout.update(self._panel)


class RichDisplay(GeneralistDisplay):
    def __init__(self, display: bool = True):
        super().__init__(display)
        self._status = "init" if display else False

    def _stop(self):
        self.live.stop()

    def run(self, debug: bool):
        if not debug:
            self.live.start()
        else:
            self.debug_run()

    def debug_run(self):
        self.sample_info.update = print

    def setup_layout(self, debug: bool = False, **kwargs):
        self.layout = Layout(name="root")

        self.layout.split(
            Layout(name="main", ratio=1),
            Layout(name="sample_info"),
        )

        self.layout_epochs = self.layout["main"]

        # if n_batches:
        #     self.layout["main"].split(
        #         Layout(name="epoch_progress", ratio=2),
        #         Layout(name="batch_progress"),
        #     )
        #     self.layout_epochs = self.layout["main"]["epoch_progress"]
        #     self.layout_batches = self.layout["main"]["batch_progress"]

        #     # self.layout_batches.update(self.setup_batch_progress(n_batches=n_batches))
        #     self.batch_progress = BatchProgress()
        #     self.layout_batches.update(self.batch_progress.panel)

        # else:
        #     self.layout_epochs = self.layout["main"]

        # self.layout_epochs.update(self.setup_epoch_progress(n_epochs=n_epochs))
        # self.epoch_progress = EpochProgress()
        # self.layout_epochs.update(self.epoch_progress.panel)

        # self.sample_info = SampleInfo(parent_layout=self.layout["sample_info"], title="Sample Info", padding=1)
        epoch_panel = self.setup_epoch_progress()
        batch_panel = self.setup_batch_progress()

        self.layout_batches.update(self.batch_progress.panel)
        self.layout_epochs.update(self.epoch_progress.panel)

        self.setup_sample_info(parent_layout=self.layout["sample_info"], title="Sample Info", padding=1)

        self.live = Live(self.layout, refresh_per_second=10, screen=True)
        atexit.register(self._stop)

        self.run(debug=debug)

    def setup_sample_info(self, parent_layout: Layout, title: str, padding=1):
        self.sample_info = SampleInfo(parent_layout=parent_layout, title=title, padding=padding)

    def setup_epoch_progress(self, n_epochs: int = 10):
        self.epoch_progress = EpochProgress()
        return self.epoch_progress.panel

    def setup_batch_progress(self, batch_iter: Iterable | int = None):
        if hasattr(self, "batch_progress"):
            print("ERROR!!!")
            # self.batch_progress.progress.reset(self.batch_progress.progress.task_ids[0])
            pass

        else:
            self.layout["main"].split(
                Layout(name="epoch_progress", ratio=2),
                Layout(name="batch_progress"),
            )

            self.layout_epochs = self.layout["main"]["epoch_progress"]
            self.layout_batches = self.layout["main"]["batch_progress"]

        self.batch_progress = BatchProgress()


    def split___(self):
        self.layout["main"].split(
            Layout(name="epoch_progress", ratio=2),
            Layout(name="batch_progress"),
        )

        self.layout_epochs = self.layout["main"]["epoch_progress"]
        self.layout_batches = self.layout["main"]["batch_progress"]


class ProgressBase:
    def __init__(self, progress_task: Iterable = None, title: str = None, task_name: str = None, **kwargs) -> None:
        # self.parent = parent
        self.progress: Progress = None

        self.progress_task = progress_task

        self.task_name = task_name if task_name is not None else self._task_name
        self.title = title if title is not None else self._task_name

    @property
    def panel(self):
        return self._panel

    def task(self, progress_task: Iterable | int):
        if isinstance(progress_task, int):
            progress_task = range(progress_task)
        self.progress_task = progress_task

        if self.progress.task_ids:
            self.progress.reset(self.progress.task_ids[0])
        else:
            self.progress.add_task(self.task_name, total=len(self.progress_task))

    def make_progress_bar(self, task_name: str, title: str = None, columns: List[ProgressColumn] = None):
        self.task_name = task_name
        self.title = title
        self.columns = columns

        self.progress = Progress(*columns)
        self._panel = Panel(self.progress, title=self.title)

    def advance(self):
        self.progress.advance(0)

    def __iter__(self):
        for i in self.progress_task:
            self.advance()
            yield i


class EpochProgress(ProgressBase):
    _task_name = "[bold blue]Epoch"
    _title = "Epoch Progress"
    _columns = [
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        SpinnerColumn("simpleDots"),
        BarColumn(bar_width=None),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.make_progress_bar(task_name=self._task_name, title=self._title, columns=self._columns)


class BatchProgress(ProgressBase):
    _task_name = "[bold blue]Batch"
    _title = "Batch Progress"
    _columns = [
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        SpinnerColumn("simpleDots"),
        BarColumn(bar_width=None),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.make_progress_bar(task_name=self._task_name, title=self._title, columns=self._columns)


if __name__ == "__main__":
    import time

    def get_args():
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true")
        return parser.parse_args()

    num_batches = 10
    batch_updates = [n for n in range(num_batches)]

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
    # breakpoint()
    display.setup_layout(debug=args.debug)
    display.epoch_progress.task(num_epochs)

    # display.run(debug=args.debug)

    # for i in range(num_epochs):
    for i in display.epoch_progress:
        display.batch_progress.task(batch_updates)
        # display.update(epoch_kwargs={"task_id": display.epoch_task, "advance": 1})
        # for ii in display.batch_progress:
        #     x = 5 - ii
        #     time.sleep(0.1)
        # for ii in range(num_batches):
        #     display.batch_progress.advance()
        #     time.sleep(0.1)

        # for ii in display.batch_progress:
        #     x = 5 - ii
        #     time.sleep(0.1)

        display.sample_info.update(sample_info_arr[i])
        time.sleep(1.5)

    display.stop()
    # display._stop()
