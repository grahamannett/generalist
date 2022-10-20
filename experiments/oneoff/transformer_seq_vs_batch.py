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


src = torch.rand((seq_len * 2, batch_size, 512)).to(device)
tgt = torch.rand((seq_len, batch_size, 512)).to(device)

src_batch_first, tgt_batch_first = src.swapaxes(0, 1), tgt.swapaxes(0, 1)


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


model_seq = Model(batch_first=False).to(device)
model_batch = Model(batch_first=True).to(device)


def tensorized_forward(model, src, target):
    return model(src, target)


t_seq = benchmark.Timer(
    stmt="tensorized_forward(model, src, tgt)",
    setup="from __main__ import tensorized_forward",
    globals={"model": model_seq, "src": src, "tgt": tgt},
)

t_batch = benchmark.Timer(
    stmt="tensorized_forward(model, src, tgt)",
    setup="from __main__ import tensorized_forward",
    globals={"model": model_batch, "src": src_batch_first, "tgt": tgt_batch_first},
)

# warmup
t_batch.timeit(1)
# print_timer("batch", t_batch.timeit(1))


print_timer("batch", t_batch.timeit(number))
print_timer("seq", t_seq.timeit(number))
