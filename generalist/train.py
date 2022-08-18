import torch
from accelerate import Accelerator
from config import config, device
from rich import print
from torch.utils.data import DataLoader

from generalist.generalist_datasets import AokvqaDataset, GeneralistDataset, CocoDataset, MNISTDataset

from generalist.generalist_tokenizers import ImageTokenizer, TextTokenizer
from generalist.data_types.input_types import ImageType, TextTypeRaw

from generalist.models.model import EmbeddingModel, GeneralistModel
from generalist.models.output_model import GeneralClassificationOutput, GeneralOutput
from generalist.predict import ImageCaptionPrediction

from generalist.utils.cli import train_get_args
from generalist.utils.display import GeneralistDisplay
from generalist.utils.utils import collate_func


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

    image_tokenizer = ImageTokenizer()
    text_tokenizer = TextTokenizer()

    embedding_model = EmbeddingModel(model_dim=model_dim)
    # output_model = GeneralClassificationOutput(model_dim=model_dim, num_classes=10, reduce_type="cls")
    output_model = GeneralOutput(model_dim=model_dim, output_dim=text_tokenizer.tokenizer.vocab_size)
    model = GeneralistModel(embedding_model=embedding_model, output_model=output_model, d_model=model_dim).to(
        device
    )

    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_model.parameters()},
            {"params": output_model.parameters()},
            {"params": model.transformer.parameters()},
            # {"params": model.parameters()},
        ],
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    TextTypeRaw.tokenizer = text_tokenizer
    ImageType.tokenizer = image_tokenizer

    tokenizers = [image_tokenizer, text_tokenizer]

    # can just call this with the base class for all datasets
    GeneralistDataset.use_tokenizers(tokenizers)
    # or can call on a specific dataset
    MNISTDataset.use_tokenizers(tokenizers)

    dataset = CocoDataset(coco_dir=config.COCO_DIR)
    out = dataset[0]

    caption_preder = ImageCaptionPrediction(text_tokenizer.tokenizer)
    caption_preder.make_caption(model, out.data.to(device), out.target.to(device))
    # breakpoint()
    # dataset = AokvqaDataset()
    # dataset = SummarizationDataset()
    # dataset = LanguageModelingDataset()

    # dataset = MNISTDataset(
    #     train=True, out_channels=3, process_sample_target=proc_target, return_raw=return_raw
    # )
    # out = dataset[0]

    # val_dataset = MNISTDataset(train=False, out_channels=3, return_raw=True)

    collate_fn = collate_func(device=device, return_target="pt")

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # out = next(iter(train_dataloader))
    # out = model(out)

    display = GeneralistDisplay.make(display=display_flag)
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

        model.train()
        image_tokenizer = ImageTokenizer()

        for batch_idx, batch in enumerate(train_dataloader):

            data, target = batch.data, batch.target
            logits = model(data)

            encoded_targets = [
                torch.nn.functional.pad(t, (0, logits.shape[1] - t.shape[-1], 0, 0), mode="constant", value=0)
                for t in target
            ]
            encoded_targets = torch.stack(encoded_targets)

            loss = loss_fn(logits.view(-1, logits.shape[-1]), encoded_targets.view(-1))

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            try:
                test_decoded = text_tokenizer.tokenizer.batch_decode(logits[:, 0].argmax(1))
                test_actual = text_tokenizer.tokenizer.batch_decode(encoded_targets[:, 0])
            except IndexError:
                breakpoint()

            batch_correct = sum([1 if a == b else 0 for a, b in zip(test_decoded, test_actual)])
            batch_total = len(test_decoded)

            running_correct += batch_correct
            running_total += batch_total

            # test_decoded = logits.argmax(dim=1)
            # test_actual = encoded_targets
            # test_decoded = logits[:, 0].argmax(1)
            # test_actual = encoded_targets[:, 0]

            # batch_correct = test_decoded.eq(test_actual).sum().item()
            # batch_total = len(test_actual)

            # running_correct += batch_correct
            # running_total += batch_total

            acc = f"{(running_correct / running_total):0.3f}"

            display_vals = {
                "acc": acc,
                "batch_acc": batch_correct / batch_total,
                "batch_idx": batch_idx,
            }

            # if batch_idx % 50 == 0:
            #     display_vals["test_decoded"] = test_decoded
            #     display_vals["test_actual"] = test_actual

            display.update(
                "batch_progress",
                advance=1,
                batch_loss=f"{loss.item():0.3f}",
                running_loss=f"{running_loss:.3f}",
                test=display_vals,
            )

        caption_preder.make_caption(model, out.data, out.target)
        break

    display.manage("epoch", display.END)
    print("done with training")


if __name__ == "__main__":
    args = train_get_args()
    train(**vars(args))
