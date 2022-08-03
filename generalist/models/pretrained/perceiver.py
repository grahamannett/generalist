import torch
from torch import nn
from transformers import PerceiverModel, PerceiverPreTrainedModel, PerceiverForMaskedLM
from transformers.models.perceiver.modeling_perceiver import PerceiverEmbeddings

from generalist.generalist_tokenizers.general_embedding import GeneralEmbedding
from generalist.models.latents import LatentEmbedding   


class PerceiverHelper(PerceiverModel):
    _keys_to_ignore_on_load_unexpected = ["perceiver"]


class TransformerDecoder(nn.Module):
    def __init__(self, pretrained_model_or_path: str = "deepmind/multimodal-perceiver", **kwargs):

        super().__init__()
        # perceiver = PerceiverHelper.from_pretrained(pretrained_model_or_path)
        perceiver_ = PerceiverForMaskedLM.from_pretrained("deepmind/language-perceiver")
        perceiver = perceiver_.perceiver
        self.config = perceiver.config
        self.encoder = perceiver.encoder

        num_latents = kwargs.get("num_latents", self.config.num_latents)
        d_latents = kwargs.get("d_latents", self.config.d_latents)

        self.embeddings = LatentEmbedding(num_latents, d_latents)

        self.model_max_length = kwargs.get("model_max_length", 2048)

    def forward(self, embedding: GeneralEmbedding, latents: torch.Tensor = None):
        if latents is None:
            latents = self.embeddings(batch_size=embedding.shape[0])

        out = self.encoder(latents, inputs=embedding.embedding)
        return out[0]


if __name__ == "__main__":
    model = TransformerDecoder("deepmind/multimodal-perceiver")
    # b x t x d
    x = torch.randn(1, 200, 704)
    out = model(x)
    breakpoint()
