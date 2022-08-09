import torch
from config import config
from torch.utils.data import DataLoader


# from tqdm import tqdm

from rich import print

from generalist.generalist_datasets.aokvqa.aokvqa import AokvqaDataset
from generalist.generalist_datasets.hf_datasets import LanguageModelingDataset, SummarizationDataset
from generalist.generalist_datasets.image_datasets import MNISTDataset

# from generalist.generalist_tokenizers.prepare_data import PrepareData
from generalist.models.pretrained.perceiver import (
    ImagePath as ImagePathPerceiver,
    PerceiverClassificationOutput,
)
from generalist.models.model import EmbeddingModel, GeneralistModel, GeneralClassificationOutput
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

    embedding_model = EmbeddingModel(model_dim=512)
    image_path_perceiver = ImagePathPerceiver()
    output_model_perceiver = PerceiverClassificationOutput()
    embedding_model.swap_data_type(module=image_path_perceiver)
    # output_model = GeneralClassificationOutput(num_classes=10)
    model = GeneralistModel(
        embedding_model=embedding_model, output_model=output_model_perceiver, d_model=512
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(
        [
            {"params": model.parameters()},
        ],
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # dataset = AokvqaDataset()
    # dataset = SummarizationDataset()
    # dataset = LanguageModelingDataset()
    dataset = MNISTDataset(train=True, out_channels=3, process_sample_target=False, return_raw=True)
    val_dataset = MNISTDataset(train=False, out_channels=3, return_raw=True)

    out = dataset[0]

    collate_fn = collate_func(device=device, return_target="pt")

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # out = next(iter(train_dataloader))
    # out = model(out)
    # breakpoint()

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

        for batch_idx, batch in enumerate(train_dataloader):

            data, target = batch.data, batch.target

            logits = model(data)

            logits_max_length = logits.shape[1]

            encoded_targets = target.to(int).to(device)
            # breakpoint()
            out = logits
            # loss = loss_fn(out.view(-1, out.shape[-1]), encoded_targets.view(-1))

            loss = loss_fn(out, encoded_targets)

            running_loss += loss.item()

            optimizer.zero_grad()
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # test_examples = out.argmax(1)[0][:5]
            # test_decoded = tokenizer.decode(test_examples)
            # test_actual = tokenizer.decode(encoded_targets[0][0])
            test_decoded = out.argmax(1)
            test_actual = encoded_targets
            running_correct += test_decoded.eq(test_actual).sum().item()
            running_total += len(test_actual)

            acc = f"{(running_correct / running_total):0.3f}"

            display.update(
                "batch_progress",
                advance=1,
                running_loss=f"{running_loss:.3f}",
                # test={"pred": test_decoded, "actual": test_actual},
                test={"pred": test_decoded, "actual": test_actual, "acc": acc, "batch_idx": batch_idx},
            )

    display.manage("epoch", display.END)
    print("done with training")


if __name__ == "__main__":
    args = train_get_args()
    train(**vars(args))
