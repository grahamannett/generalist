import torch

def pad_targets(targets, logits):
    # pad targets to match logits
    encoded_targets = [torch.nn.functional.pad(t, (0, logits.shape[1] - t.shape[-1], 0, 0), mode="constant", value=0) for t in targets]
    encoded_targets = torch.stack(encoded_targets)
