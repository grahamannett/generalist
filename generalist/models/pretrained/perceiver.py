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

from generalist.generalist_embedding.general_embedding import GenearlizedTensor
from generalist.models.latents import LatentEmbedding


class PerceiverHelper(PerceiverModel):
    _keys_to_ignore_on_load_unexpected = ["perceiver"]


class PerceiverClassificationOutput(nn.Module):
    def __init__(self, num_classes: int = 10, **kwargs) -> None:
        super().__init__()
        model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")
        self.decoder = model.perceiver.decoder

        self.decoder.decoder.final_layer = nn.LazyLinear(num_classes)

        linear_hidden_shape = list(self.decoder.decoder.output_position_encodings.parameters())[0].shape[-1]
        self.linear_hidden = nn.LazyLinear(linear_hidden_shape)

    def forward(self, hidden_states: torch.Tensor, decoder_query: torch.Tensor, **kwargs) -> torch.Tensor:

        query = self.decoder.decoder_query(decoder_query)
        sequence_output = self.linear_hidden(hidden_states[:, 0])
        out = self.decoder(query, z=sequence_output.unsqueeze(1))
        return out[0]


class ImagePath(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_type = "image"
        model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")
        config = model.config

        # pretrained
        input_preprocessor = model.perceiver.input_preprocessor
        latents = model.perceiver.embeddings
        # decoder = model.perceiver.decoder

        self.embedding_path = input_preprocessor

    def forward(self, data: GenearlizedTensor) -> GenearlizedTensor:
        if data.data.ndim == 3:
            data.data = data.data.unsqueeze(0)
        embedding = self.embedding_path(data.data)[0]
        embedding = GenearlizedTensor(embedding).set_data_type(self.data_type)
        return embedding


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
