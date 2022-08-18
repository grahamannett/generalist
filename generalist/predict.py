import torch


class ImageCaptionPrediction:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def make_caption(self, model, tokenized_image, tokenized_caption):
        logits = model([tokenized_image])
        generated_caption = logits.argmax(1)[:, : tokenized_caption.shape[-1]]
        normal_caption, generated_caption = self.tokenizer.batch_decode(
            torch.cat((tokenized_caption, generated_caption))
        )
        print(f"generated caption:\n==>{generated_caption}")
        print(f"actual caption:\n==>{normal_caption}")
        breakpoint()
