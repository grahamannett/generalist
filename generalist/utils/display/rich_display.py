import atexit
from datasets import load_dataset
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
from generalist.generalist_datasets.hf.summary import SummaryTransforms, XSum
from generalist.generalist_datasets.utils.data_collate import collate_func_helper
from generalist.generalist_tokenizers import text_tokenizers

from generalist.utils.display.display import GeneralistDisplay


class SampleInfo:
    TableCls = Table  # or Table.grid

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

        # self.table = Table.grid(padding=self.padding)
        self.table = self.TableCls(padding=self.padding)

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


class RichDisplay:
    def __init__(self, display: bool = True, **kwargs):
        # super().__init__(display)
        self._status = "init" if display else False

    def extra_setup(self, *args, **kwargs):
        pass

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

        self.setup_epoch_progress()
        self.setup_batch_progress()

        self.layout_epochs.update(self.epoch_progress.panel)
        self.layout_batches.update(self.batch_progress.panel)

        self.setup_sample_info(parent_layout=self.layout["sample_info"], title="Sample Info", padding=1)

        self.live = Live(self.layout, refresh_per_second=10, screen=True)
        atexit.register(self._stop)

        self.run(debug=debug)

    def setup_sample_info(self, parent_layout: Layout, title: str, padding=1):
        self.sample_info = SampleInfo(parent_layout=parent_layout, title=title, padding=padding)

    def setup_epoch_progress(self, n_epochs: int = 10):
        self.epoch_progress = EpochProgress()
        # self.layout_epochs = self.layout["main"]
        return self.epoch_progress.panel

    def setup_batch_progress(self, batch_iter: Iterable | int = None):
        if hasattr(self, "batch_progress"):
            raise NotImplementedError("batch progress already set")
        #     pass
        # else:

        self.batch_progress = BatchProgress()
        self.layout["main"].split(
            Layout(name="epoch_progress", ratio=2),
            Layout(name="batch_progress"),
        )

        self.layout_epochs = self.layout["main"]["epoch_progress"]
        self.layout_batches = self.layout["main"]["batch_progress"]

    def split___(self):
        self.layout["main"].split(
            Layout(name="epoch_progress", ratio=2),
            Layout(name="batch_progress"),
        )

        self.layout_epochs = self.layout["main"]["epoch_progress"]
        self.layout_batches = self.layout["main"]["batch_progress"]

    def update(self, *args):
        self.sample_info.update(*args)


class ProgressBase:
    def __init__(
        self, task_name: str = None, title: str = None, columns: List[ProgressColumn] = None, progress_task: Iterable = None, **kwargs
    ) -> None:
        self.progress: Progress = None

        self.progress_task = progress_task

        self.task_name = task_name if task_name is not None else self._task_name
        self.title = title if title is not None else self._task_name
        self.columns = columns

        if task_name and title and columns:
            self.make_progress_bar(task_name=self.task_name, title=self.title, columns=self.columns)

    @property
    def panel(self):
        return self._panel

    def task(self, progress_task: Iterable | int, make_enumerate: bool = False):
        if isinstance(progress_task, int):
            progress_task = range(progress_task)

        self.progress_task = progress_task

        # if self.progress.task_ids:
        #     self.progress.reset(self.progress.task_ids[0])
        # else:
        # self.progress.add_task(self.task_name, total=len(self.progress_task))
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
        super().__init__(task_name=self._task_name, title=self._title, columns=self._columns, **kwargs)


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
        super().__init__(task_name=self._task_name, title=self._title, columns=self._columns, **kwargs)


if __name__ == "__main__":
    import time

    def get_args():
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true")
        return parser.parse_args()

    num_batches = 10
    batch_updates = [n for n in range(num_batches)]

    def make_obj(idx: int = 5):
        return {"key1": "val1", "key2": "val2", "key3": "val3", "key4": "val4", "key5": "val5", "key6": "val6"}

    sample_info_arr = [
        [
            {"idx": 1, "text": "This is a text1", "obj": make_obj()},
            {"idx": 1, "text": "This is a text1", "obj": make_obj()},
            {"idx": 1, "text": "This is a text1", "obj": make_obj()},
        ],
        [{"idx": 2, "text": "This is a text2", "obj": make_obj()}, {"idx": 2, "text": "This is a text2", "obj": make_obj()}],
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
    from datasets import load_dataset

    debug = args.debug
    # xsum = load_dataset("xsum", split="train")
    text_tokenizer = text_tokenizers.TextTokenizerBert.from_pretrained("bert-base-uncased")
    text_tokenizer_kwargs = {}
    xsum = XSum(
        text_transform=SummaryTransforms.make_transforms(text_tokenizer=text_tokenizer, text_tokenizer_kwargs=text_tokenizer_kwargs).train,
    )
    out = xsum[0]

    collate_fn = collate_func_helper(device="cpu", return_tensors="pt")
    dataloader = DataLoader(xsum, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # for x in dataloader:
    #     breakpoint()

    # breakpoint()
    display = RichDisplay()
    num_epochs = len(sample_info_arr)
    # breakpoint()
    display.setup_layout(debug=args.debug)
    display.epoch_progress.task(num_epochs)

    for i in display.epoch_progress:
        display.batch_progress.task(dataloader)

        for ii in display.batch_progress:
            # time.sleep(0.1)
            pass

        # display.sample_info.update(sample_info_arr[i])
        display.update(sample_info_arr[i])
        time.sleep(1)

    display.stop()
    # display._stop()
