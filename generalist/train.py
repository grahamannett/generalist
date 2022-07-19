from argparse import Namespace
from base64 import encode
import torch
from config import config
from torch.utils.data import DataLoader
from torchvision.io import read_image

# from tqdm import tqdm

from rich import print

from generalist.generalist_datasets.aokvqa.aokvqa import AokvqaDataset
from generalist.generalist_datasets.hf_datasets import LanguageModelingDataset, SummarizationDataset
from generalist.generalist_datasets.image_datasets import MNISTDataset
from generalist.generalist_tokenizers.prepare_data import PrepareData
from generalist.models.model import EmbeddingModel, GeneralistModel
from generalist.utils.utils import sample_collate_fn, get_args

device = config.device


from generalist.utils.display import GeneralistDisplay


def train_step(embedding_model, genearlist_model, dataloader):
    pass


def manage_live(group):
    pass


def train(**kwargs):

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

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=sample_collate_fn)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, collate_fn=sample_collate_fn
    )



    # with Live(progress_group):

    # live_rich.start(True)

    display_flag = kwargs.get("display", True)
    display = GeneralistDisplay(Live(progress_group), display=display_flag)

    display.manage()
    for epoch in range(n_epochs):
        # epoch_progress.update(epoch_task)

        running_loss = 0.0
        display.add_task("batch_task", display.batch_progress, batch_kwargs=)
        batch_task = batch_progress.add_task(
            "[green]Batch", total=len(train_dataloader), running_loss=running_loss
        )

        for idx, batch in enumerate(train_dataloader):

            data, target = batch

            data_tokenized = prepare_data(data)
            data_embedded = embedding_model(data_tokenized)
            logits = model(data_embedded)

            encoded_targets = prepare_data.prepare_targets(target, logits)

            out = torch.cat(logits, dim=1).squeeze(0)

            # encoded_targets = torch.cat(encoded_targets, dim=1).squeeze(0).to(device)
            encoded_targets = encoded_targets.reshape(-1).to(device)

            if len(out) != len(encoded_targets):
                breakpoint()

            loss = loss_fn(out, encoded_targets)

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            batch_progress.update(
                batch_task,
                advance=1,
                running_loss=running_loss,
            )
            break
        break

    display.manage()
    print("done with training")


if __name__ == "__main__":
    args = get_args()
    train(**vars(args))
