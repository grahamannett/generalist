import torch
from config import config
from torch.utils.data import DataLoader


# from tqdm import tqdm

from rich import print

from generalist.generalist_datasets.aokvqa.aokvqa import AokvqaDataset
from generalist.generalist_datasets.hf_datasets import LanguageModelingDataset, SummarizationDataset
from generalist.generalist_datasets.image_datasets import MNISTDataset
from generalist.generalist_tokenizers.image_tokenizer import ImageTokenizer
from generalist.generalist_tokenizers.text_path import TextTokenizer

# from generalist.generalist_tokenizers.prepare_data import PrepareData
from generalist.models.pretrained.perceiver import (
    ImagePath as ImagePathPerceiver,
    PerceiverClassificationOutput,
)
from generalist.models.model import (
    EmbeddingModel,
    GeneralOutput,
    GeneralistModel,
    GeneralClassificationOutput,
)
from generalist.utils.utils import Batch, sample_collate_fn, collate_func
from generalist.utils.cli import train_get_args

from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device


from generalist.utils.display import GeneralistDisplay


def train_step(embedding_model, genearlist_model, dataloader):
    pass


def manage_live(group):
    pass


def train(**kwargs):
    lr = kwargs.get("lr", 5e-5)
    n_epochs = kwargs.get("n_epochs", 1)
    batch_size = kwargs.get("batch_size", 1)
    display_flag = kwargs.get("display", True)
    model_dim = kwargs.get("model_dim", 768)

    embedding_model = EmbeddingModel(model_dim=model_dim)
    # output_model = GeneralClassificationOutput(model_dim=model_dim, num_classes=10, reduce_type="cls")
    output_model = GeneralOutput(model_dim=model_dim)
    model = GeneralistModel(embedding_model=embedding_model, output_model=output_model, d_model=model_dim).to(
        device
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            # {"params": embedding_model.parameters()},
            # {"params": output_model.parameters()},
            # {"params": model.transformer.parameters()},
            {"params": model.parameters()},
        ],
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    image_tokenizer = ImageTokenizer()
    text_tokenizer = TextTokenizer()
    tokenizers = [image_tokenizer, text_tokenizer]

    # dataset = AokvqaDataset()
    # dataset = SummarizationDataset()
    # dataset = LanguageModelingDataset()
    proc_target = True
    return_raw = False

    MNISTDataset.use_tokenizers(tokenizers)

    dataset = MNISTDataset(
        train=True, out_channels=3, process_sample_target=proc_target, return_raw=return_raw
    )

    val_dataset = MNISTDataset(train=False, out_channels=3, return_raw=True)

    out = dataset[0]

    collate_fn = collate_func(device=device, return_target="pt")

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # out = next(iter(train_dataloader))
    # out = model(out)

    display = GeneralistDisplay.make(display=display_flag)
    display.manage()

    model, optimizer, data = accelerator.prepare(model, optimizer, train_dataloader)

    for epoch in range(n_epochs):
        # epoch_progress.update(epoch_task)

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        display.update("epoch_progress", epoch)
        display.add_task(
            "batch_progress", "[green]Batch", total=len(train_dataloader), running_loss=running_loss
        )

        model.train()
        image_tokenizer = ImageTokenizer()

        for batch_idx, batch in enumerate(train_dataloader):

            data, target = batch.data, batch.target

            logits = model(data)

            logits_max_length = logits.shape[1]

            encoded_targets = torch.stack(target).squeeze(1).to(int).to(device)
            # encoded_targets = target.to(int).to(device)

            # breakpoint()
            loss = loss_fn(logits[:, 0], encoded_targets[:, 0])
            # loss = loss_fn(logits, encoded_targets)

            running_loss += loss.item()

            optimizer.zero_grad()
            # accelerator.backward(loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            test_decoded = logits[:, 0].argmax(1)
            test_actual = encoded_targets[:, 0]

            batch_correct = test_decoded.eq(test_actual).sum().item()
            batch_total = len(test_actual)

            running_correct += batch_correct
            running_total += batch_total

            acc = f"{(running_correct / running_total):0.3f}"

            display_vals = {
                "acc": acc,
                "batch_acc": batch_correct / batch_total,
                "batch_idx": batch_idx,
            }

            display.update(
                "batch_progress",
                advance=1,
                running_loss=f"{running_loss:.3f}",
                test=display_vals,
            )

    display.manage("epoch", display.END)
    print("done with training")


if __name__ == "__main__":
    args = train_get_args()
    train(**vars(args))
