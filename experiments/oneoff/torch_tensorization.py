import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import sys


device = "cuda"
number = int(sys.argv[1]) if len(sys.argv) > 1 else 1
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 64

transformer_kwargs = {
    "nhead": 16,
    "num_encoder_layers": 12,
    "batch_first": True,
}


def print_timer(name: str, timer: benchmark.utils.common.Measurement):
    print(f"\n==>{name}: \n{timer}\n=== ===\n")


src = torch.rand((64, 64, 512)).to(device)
tgt = torch.rand((64, 64, 512)).to(device)

# src_seq_first = torch.rand((10, 32, 512)).to(device)
# tgt_seq_first = torch.rand((20, 32, 512)).to(device)


# class ImageModel(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()


# class Transformer(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         default_params = {
#             **transformer_kwargs,
#             **kwargs,
#         }
#         self.transformer = nn.Transformer(**kwargs)

#     def forward(self, src, target, modality: str = None):
#         return self.transformer(src, target)


class Model(nn.Module):
    def __init__(self, batch_first: bool = None) -> None:
        super().__init__()

        self.transformer_kwargs_ = {**transformer_kwargs}

        if batch_first is not None:
            self.transformer_kwargs_["batch_first"] = batch_first

        self.embedding = nn.Linear(512, 512)
        self.transformer = nn.Transformer(**self.transformer_kwargs_)

    def forward(self, src, tgt, *args, **kwargs):
        src = self.embedding(src)
        return self.transformer(src, tgt)


class ModelAlt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.ModuleDict({"base": nn.Linear(512, 512)})
        self.transformer = nn.Transformer(**transformer_kwargs)

    def batch_embedding(self, src, modality):
        return self.embedding[modality](src)

    def forward(self, src, tgt, modality, *args, **kwargs):
        src = self.embedding[modality](src)
        return self.transformer(src, tgt)


model = Model().to(device)
model_alt = ModelAlt().to(device)


def tensorized_forward(model, src, target, modality: str = None, device: str = "cuda"):
    # src, target = src.to(device), target.to(device)
    return model(src, target)


# def single_forward(model, src, target, modality: str = None, device: str = "cuda"):
#     src, target = src.to(device), target.to(device)

#     out = []
#     for i in range(32):
#         out.append(model(src[i : i + 1, :], target[i : i + 1, :], modality))

#     return out


# t0 = benchmark.Timer(
#     stmt="tensorized_forward(model, src, tgt)",
#     setup="from __main__ import tensorized_forward",
#     globals={"model": model, "src": src, "tgt": tgt},
# )

# t0 = benchmark.Timer(
#     stmt="tensorized_forward(model, src, tgt, 'cuda')",
#     setup="from __main__ import tensorized_forward",
#     globals={"model": model, "src": src, "tgt": tgt},
# )

t_seq = benchmark.Timer(
    stmt="tensorized_forward(model, src, tgt, 'cuda')",
    setup="from __main__ import tensorized_forward",
    globals={"model": model_seq, "src": src, "tgt": tgt},
)
t_batch = benchmark.Timer(
    stmt="tensorized_forward(model, src, tgt, 'cuda')",
    setup="from __main__ import tensorized_forward",
    globals={"model": model_batch, "src": src, "tgt": tgt},
)

print_timer("seq", t_seq.timeit(number))
print_timer("batch", t_batch.timeit(number))

# t1 = benchmark.Timer(
#     stmt="single_forward(model, src, tgt, modality, 'cuda')",
#     setup="from __main__ import single_forward",
#     globals={"model": model_alt, "src": src, "tgt": tgt, "modality": "base"},
# )

# print(t0.timeit(10))
# print_timer("t0", t0.timeit(number))
# print_timer("t1", t1.timeit(number))
