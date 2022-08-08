import torch
from torch import nn
from transformers import (
    PerceiverModel,
    PerceiverPreTrainedModel,
    PerceiverForMaskedLM,
)
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverEmbeddings,
    PerceiverImagePreprocessor,
    PerceiverForImageClassificationLearned,
)

from generalist.generalist_tokenizers.general_embedding import GenearlizedTensor
from generalist.models.latents import LatentEmbedding


class PerceiverHelper(PerceiverModel):
    _keys_to_ignore_on_load_unexpected = ["perceiver"]


class ImagePath(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_type = "image"
        config = PerceiverForImageClassificationLearned.from_pretrained(
            "deepmind/vision-perceiver-learned"
        ).config

        trainable_position_encoding_kwargs_preprocessor = {
            "num_channels": 256,
            "index_dims": config.image_size**2,
        }
        input_preprocessor = PerceiverImagePreprocessor(
            config=config,
            prep_type="conv1x1",
            spatial_downsample=1,
            out_channels=256,
            position_encoding_type="trainable",
            concat_or_add_pos="concat",
            project_pos_dim=256,
            trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_preprocessor,
        )

        self.embedding = input_preprocessor

    def forward(self, data: GenearlizedTensor) -> GenearlizedTensor:
        if data.data.ndim == 3:
            data.data = data.data.unsqueeze(0)
        out = GenearlizedTensor(self.embedding(data.data)[0]).set_data_type(self.data_type)
        return out


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

    def forward(self, embedding: GenearlizedTensor, latents: torch.Tensor = None):
        if latents is None:
            latents = self.embeddings(batch_size=embedding.shape[0])

        out = self.encoder(latents, inputs=embedding.embedding)
        return out[0]


if __name__ == "__main__":
    # model = TransformerDecoder("deepmind/multimodal-perceiver")
    # # b x t x d
    # x = torch.randn(1, 200, 704)
    # out = model(x)

    x = torch.rand(2, 3, 224, 224)
    out = ImagePath()(x)
    breakpoint()
