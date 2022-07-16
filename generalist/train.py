from base64 import encode
import torch
from config import config
from torch.utils.data import DataLoader
from torchvision.io import read_image

# from tqdm import tqdm
# from rich.progress import track
from rich import print

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


from generalist.generalist_datasets.aokvqa.aokvqa import AokvqaDataset
from generalist.generalist_datasets.hf_datasets import LanguageModelingDataset, SummarizationDataset
from generalist.generalist_datasets.image_datasets import MNISTDataset
from generalist.generalist_tokenizers.prepare_data import PrepareData
from generalist.models.model import EmbeddingModel, GeneralistModel
from generalist.utils.utils import sample_collate_fn, get_args

device = config.device


def train():

    args = get_args()
    #

    lr = 5.0  # learning rate
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    embedding_model = EmbeddingModel().to(device)
    model = GeneralistModel().to(device)

    prepare_data = PrepareData(embedding_model=embedding_model, generalist_model=model, device=device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        [
            {"params": embedding_model.parameters()},
            {"params": model.parameters()},
        ],
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # dataset = AokvqaDataset()
    # dataset = SummarizationDataset()
    # dataset = LanguageModelingDataset()
    dataset = MNISTDataset(train=True)
    val_dataset = MNISTDataset(train=False)

    _ = dataset[0]
    # breakpoint()
    # breakpoint()

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=sample_collate_fn)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, collate_fn=sample_collate_fn
    )

    epoch_progress = Progress("{task.description}", MofNCompleteColumn(), BarColumn())
    batch_progress = Progress(
        "{task.description}",
        MofNCompleteColumn(),
        SpinnerColumn(),
        # BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )

    progress_group = Group(
        Panel(epoch_progress),
        Panel(Group(batch_progress)),
    )

    epoch_task = epoch_progress.add_task("[blue]Epoch", total=n_epochs)

    with Live(progress_group):
        for epoch in range(n_epochs):
            epoch_progress.update(epoch_task)

            # print(f"on epoch: {epoch}")
            batch_task = batch_progress.add_task("[green]Batch", total=len(train_dataloader))

            # pbar = tqdm(train_dataloader)
            running_loss = 0.0
            # pbar.set_description(f"Epoch {epoch}|{n_epochs}")
            # for idx, batch in enumerate(pbar):
            # for idx, batch in track(enumerate(train_dataloader), total=len(train_dataloader)):
            for idx, batch in enumerate(train_dataloader):

                batch_progress.update(
                    batch_task,
                    advance=1,
                )

                data, target = batch

                # the multi step process but explicit

                data_tokenized = prepare_data(data)
                data_embedded = embedding_model(data_tokenized)
                logits = model(data_embedded)

                encoded_targets = prepare_data.prepare_targets(target, logits)

                out = torch.cat(logits, dim=1).squeeze(0)
                encoded_targets = torch.cat(encoded_targets, dim=1).squeeze(0).to(device)

                if len(out) != len(encoded_targets):
                    breakpoint()

                loss = loss_fn(out, encoded_targets)

                running_loss += loss.item()
                # pbar.set_postfix(loss=f"{running_loss:.3f}")

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()


if __name__ == "__main__":

    train()
