from typing import Any, Callable, Dict, Sequence
import torch
import hydra
from generalist.data_types.helper_types import Sample
from generalist.generalist_datasets.coco.coco import (
    CocoCaption,
    CocoCaptionTargetTranform,
    CocoFilepaths,
    CocoImageTransforms,
)
from generalist.generalist_datasets.coco.eval import CocoEval
from generalist.generalist_tokenizers import image_tokenizers, text_tokenizers
from generalist.models.embedding_model import EmbeddingModel
from generalist.models.model import GeneralistModel

from generalist.utils.utils import get_hostname
from omegaconf import DictConfig

from generalist.predict import ImageCaptionPrediction


def get_max_length(arr: torch.Tensor) -> int:
    return torch.where(arr == 0)[1][0].item()


def make_out_target_true(target: torch.Tensor, text_tokenizer: text_tokenizers.TextTokenizer, max_length: int = 32) -> str:
    max_length = get_max_length(target)
    out_target_true = text_tokenizer.decode(target[0, :max_length])
    return out_target_true


def preliminary_eval(
    sample: Sample,
    text_tokenizer: text_tokenizers.TextTokenizer,
    image_tokenizer: image_tokenizers.ImageTokenizer,
    embedding_model: EmbeddingModel,
    model: GeneralistModel,
    device: str,
    initial_caption: bool,
) -> Sequence[Sequence[int]]:
    sample_data = sample.data
    out_target_tokens = sample.target

    max_length = get_max_length(sample.target)
    out_target_true = make_out_target_true(out_target_tokens, text_tokenizer, max_length=max_length)

    caption_preder = ImageCaptionPrediction(
        image_tokenizer=image_tokenizer, text_tokenizer=text_tokenizer, embedding_model=embedding_model, model=model, device=device
    )

    # tokenized_caption = out_target_tokens.to(device)
    generated_captions = []

    if initial_caption:
        initial_caption = caption_preder.make_caption(
            data=sample_data,
            max_length=max_length,
        )
        generated_captions.append(initial_caption)
    return generated_captions, caption_preder, out_target_true


def eval_predictions(cfg: DictConfig):
    pass

    # def on_job_end(self, config: DictConfig, **kwargs: Any) -> None:
    #     print(f"Job ended,uploading...")
    #     # uploading...


@hydra.main(config_path=f"../config", config_name=get_hostname(), version_base=None)
def eval(cfg: DictConfig):
    device = cfg.device
    # command = cfg.eval.command

    # coco_dataset = CocoDataset(coco_dir=cfg.coco_dir, device=device)
    coco_val_dataset = CocoCaption(coco_dir=cfg.coco_dir, device=device, split="val")

    obj = torch.load(cfg.model_save_dir + "/latest.pt")

    model = obj["model"]
    embedding_model = obj["embedding_model"]
    text_tokenizer = obj["tokenizers"]["text_tokenizer"]
    image_tokenizer = obj["tokenizers"]["image_tokenizer"]
    coco_val_dataset.use_tokenizers([text_tokenizer, image_tokenizer])

    caption_preder = ImageCaptionPrediction(
        image_tokenizer=image_tokenizer, text_tokenizer=text_tokenizer, embedding_model=embedding_model, model=model, device=device
    )

    eval_coco = CocoEval(coco_val_dataset, caption_preder)

    baseline_results = eval_coco.make_captions_baseline()
    pred_results = eval_coco.make_captions_prediction()

    annotation_file = str(coco_val_dataset.captions_path)
    baseline_results_file = "experiments/results/coco_val_baseline.json"
    pred_results_file = "experiments/results/coco_pred_results.json"
    eval_coco.save_results(baseline_results, baseline_results_file)
    eval_coco.save_results(pred_results, pred_results_file)

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # annotation_file = 'captions_val2014.json'
    # results_file = 'captions_val2014_fakecap_results.json'

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(baseline_results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params["image_id"] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")


def main():
    pass


class Commands:
    eval = eval_predictions


if __name__ == "__main__":
    eval()
