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
from generalist.utils.utils import Batch, sample_collate_fn, get_args

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
    # model = GeneralistModel().to(device)
    model = GeneralistModel(output_dim=10).to(device)

    prepare_data = PrepareData(embedding_model=embedding_model, generalist_model=model, device=device)
    tokenizer = prepare_data.tokenizer

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
    dataset = MNISTDataset(train=True, out_channels=3)
    val_dataset = MNISTDataset(train=False, out_channels=3)

    _ = dataset[0]

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.collate_fn)

    display = GeneralistDisplay.make(display=kwargs.get("display", True))
    display.manage()

    for epoch in range(n_epochs):
        # epoch_progress.update(epoch_task)

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        display.update("epoch_progress", epoch)
        display.add_task(
            "batch_progress", "[green]Batch", total=len(train_dataloader), running_loss=running_loss
        )

        for idx, batch in enumerate(train_dataloader):

            data, target = batch.data, batch.target

            data_tokenized = prepare_data(data)

            data_embedded = embedding_model(data_tokenized)

            # this wont work if there are different sizes of data (e.g. different sequence lengths)
            data_embedded = torch.vstack([d.embedding for d in data_embedded])
            logits = model(data_embedded)

            # logits_max_length = max((l.shape[1] for l in logits))
            logits_max_length = logits.shape[1]
            # encoded_targets = prepare_data.prepare_targets(target, logits_max_length=logits_max_length)

            # encoded_targets = tokenizer(
            #     [t.data for t in target],
            #     return_tensors="pt",
            #     padding="max_length",
            #     max_length=logits.shape[1],
            # )["input_ids"]
            # max_length=logits_max_length,
            # padding=padding,
            # truncation=truncation,
            # )
            # encoded_targets = encoded_targets["input_ids"]
            encoded_targets = torch.Tensor([int(t.data) for t in target]).to(int)
            # breakpoint()
            out = logits
            # loss = loss_fn(out.view(-1, out.shape[-1]), encoded_targets.view(-1))
            loss = loss_fn(out, encoded_targets)

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

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
                test={"pred": test_decoded, "actual": test_actual, "acc": acc},
            )
            # break
        break

    display.manage("epoch", display.END)
    print("done with training")


if __name__ == "__main__":
    args = get_args()
    train(**vars(args))
