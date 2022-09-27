import torch
import hydra
from generalist.generalist_datasets import CocoDataset
from generalist.utils.utils import get_hostname
from omegaconf import DictConfig

from generalist.predict import ImageCaptionPrediction


@hydra.main(config_path=f"../config", config_name=get_hostname(), version_base=None)
def eval(cfg: DictConfig):
    device = cfg.device
    coco_dataset = CocoDataset(coco_dir=cfg.coco_dir, device=device)

    obj = torch.load(cfg.model_save_dir + "/latest.pt")

    image_tokenizer, text_tokenizer = obj["tokenizers"]["image_tokenizer"], obj["tokenizers"]["text_tokenizer"]
    coco_dataset.use_tokenizers([image_tokenizer, text_tokenizer])

    model = obj["model"]
    embedding_model = obj["embedding_model"]

    caption_preder = ImageCaptionPrediction(image_tokenizer=image_tokenizer, text_tokenizer=text_tokenizer)

    out = coco_dataset[0]
    breakpoint()
    tokenized_image = out.data.to(device)
    tokenized_caption = out.target.to(device)
    # exit()

    captions = []
    captions.append(caption_preder.make_caption(embedding_model, model, tokenized_image, tokenized_caption, use_caption=False))
    captions.append(caption_preder.make_caption(embedding_model, model, tokenized_image, tokenized_caption, use_caption=True))

    decoded = text_tokenizer.batch_decode(captions)

    breakpoint()


if __name__ == "__main__":
    eval()
