import datetime
import logging
from pathlib import Path
from random import sample

import hydra
import torch
from omegaconf import DictConfig
from rich import print
from torch.utils.data import DataLoader

from generalist.data_types.input_types import ImageType, TextTypeRaw
from generalist.generalist_datasets import AokvqaDataset, CocoDataset, GeneralistDataset, MNISTDataset
from generalist.generalist_datasets.base import ChainedGenearlistDataset
from generalist.generalist_tokenizers import image_tokenizers, text_tokenizers
from generalist.models.model import EmbeddingModel, GeneralistModel
from generalist.models.output_model import GeneralClassificationOutput, GeneralOutput
from generalist.predict import ImageCaptionPrediction
from generalist.utils.display import GeneralistDisplay
from generalist.utils.utils import collate_func, get_hostname, save_checkpoint

log = logging.getLogger(__name__)


def train_step(embedding_model, genearlist_model, dataloader):
    pass


def manage_live(group):
    pass


def pad_targets(targets, logits):
    # pad targets to match logits
    encoded_targets = [torch.nn.functional.pad(t, (0, logits.shape[1] - t.shape[-1], 0, 0), mode="constant", value=0) for t in targets]
    encoded_targets = torch.stack(encoded_targets)


@hydra.main(config_path=f"../config", config_name=get_hostname(), version_base=None)
def train(cfg: DictConfig):

    # print("Working directory : {}".format(os.getcwd()))
    model_save_dir = Path(cfg.model_save_dir)
    display_flag = cfg.display
    device = cfg.device

    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    n_epochs = cfg.training.n_epochs

    model_dim = cfg.model.model_dim

    image_tokenizer = image_tokenizers.ImageTokenizer(device=device)
    text_tokenizer = text_tokenizers.TextTokenizerBert.from_pretrained("bert-base-uncased")

    # text_tokenizer = TextTokenizerPretrained("BertTokenizer", pretrained_name_or_model="bert-base-uncased", device=device)

    embedding_model = EmbeddingModel(model_dim=model_dim)
    # output_model = GeneralClassificationOutput(model_dim=model_dim, num_classes=10, reduce_type="cls")
    output_model = GeneralOutput(model_dim=model_dim, output_dim=text_tokenizer.vocab_size)
    model = GeneralistModel(output_model=output_model, d_model=model_dim).to(device)

    embedding_model.to(device)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_model.parameters()},
            {"params": model.parameters()},
            # {"params": model.transformer.parameters()},
            # {"params": output_model.parameters()},
        ],
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    TextTypeRaw.tokenizer = text_tokenizer
    ImageType.tokenizer = image_tokenizer

    tokenizers = [image_tokenizer, text_tokenizer]

    # can just call this with the base class for all datasets
    GeneralistDataset.use_tokenizers(tokenizers)
    # or can call on a specific dataset
    MNISTDataset.use_tokenizers(tokenizers)

    coco_dataset = CocoDataset(coco_dir=cfg.coco_dir, device=device)

    dataset = coco_dataset
    out = dataset[0]

    # out.data = out.data.to(device)
    # out.target = out.target.to(device)
    caption_preder = ImageCaptionPrediction(image_tokenizer=image_tokenizer, text_tokenizer=text_tokenizer)

    tokenized_image = out.data.to(device)
    tokenized_caption = out.target.to(device)
    # exit()

    inital_caption = caption_preder.make_caption(
        embedding_model=embedding_model,
        model=model,
        tokenized_image=tokenized_image,
        tokenized_caption=tokenized_caption,
    )

    generated_captions = []
    generated_captions.append(inital_caption)

    # captions_out = caption_preder.make_caption(embedding_model, model, out.data, out.target)
    # captions_info[-1] = captions_out["normal"]
    # captions_info[0] = captions_out["generated"]

    # dataset = AokvqaDataset()
    # dataset = SummarizationDataset()
    # dataset = LanguageModelingDataset()

    # dataset = MNISTDataset(train=True, out_channels=3)

    # datasets = ChainedGenearlistDataset(datasets=[dataset])

    # out = dataset[0]
    # breakpoint()

    # val_dataset = MNISTDataset(train=False, out_channels=3, return_raw=True)
    # collate_fn = collate_func(device=device, return_tensors="pt")
    collate_fn = collate_func(device=device)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    display = GeneralistDisplay.make(display=display_flag, logger=log)
    display.manage()
    for epoch in range(n_epochs):
        # epoch_progress.update(epoch_task)

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        display.update("epoch_progress", epoch)
        display.add_task("batch_progress", "[green]Batch", total=len(train_dataloader), running_loss=running_loss)

        model.train()

        for batch_idx, batch in enumerate(train_dataloader):

            data, target = batch.data, batch.target

            embedding = embedding_model(data)
            embedded_target = embedding_model(target)

            encoded_target = torch.cat(target)

            logits = model(embedding, embedded_target)

            loss = loss_fn(logits.view(-1, logits.shape[-1]), encoded_target.view(-1))

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            try:
                test_decoded = text_tokenizer.batch_decode(logits.argmax(dim=-1))
                test_actual = text_tokenizer.batch_decode(encoded_target)

            except IndexError:
                breakpoint()

            batch_correct = sum([1 if a == b else 0 for a, b in zip(test_decoded, test_actual)])
            batch_total = len(test_decoded)

            running_correct += batch_correct
            running_total += batch_total

            # breakpoint()
            if batch_idx % 50 == 0:
                decoded__ = text_tokenizer.batch_decode(logits.argmax(dim=-1)[0:5, 0:10])
                actual__ = text_tokenizer.batch_decode(encoded_target[0:5, 0:10])
                print(list(zip(decoded__, actual__)))

            acc = f"{(running_correct / running_total):0.3f}"

            display_vals = {
                "acc": acc,
                "batch_acc": batch_correct / batch_total,
                "batch_idx": batch_idx,
            }

            # if batch_idx % 50 == 0:
            #     display_vals["test_decoded"] = test_decoded[0][:75]
            #     display_vals["test_actual"] = test_actual[0][:75]

            display.update(
                "batch_progress",
                advance=1,
                batch_loss=f"{loss.item():0.3f}",
                running_loss=f"{running_loss:.3f}",
                test=display_vals,
            )
            # break

        # save_checkpoint(model_save_dir, model, embedding_model, optimizer, epoch)
        save_checkpoint(
            model_save_dir,
            obj={
                "model": model,
                "embedding_model": embedding_model,
                "optimizer": optimizer,
                "epoch": epoch,
                "tokenizers": {
                    "image_tokenizer": image_tokenizer,
                    "text_tokenizer": text_tokenizer,
                },
            },
            filename="latest.pt",
        )

        latest_caption = caption_preder.make_caption(embedding_model, model, tokenized_image, tokenized_caption, use_caption=False)
        generated_captions.append(latest_caption)

    captions_generated = text_tokenizer.batch_decode(generated_captions)

    display.manage("epoch", display.END)
    print("done with training")


if __name__ == "__main__":
    train()
