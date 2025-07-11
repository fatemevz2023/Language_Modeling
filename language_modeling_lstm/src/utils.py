### src/utils.py

import torch

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def data_process(raw_text_iter, vocab, seq_len):
    """Converts raw text iterator to input-target sequences for language modeling."""
    from torchtext.data.utils import get_tokenizer
    tokenizer = get_tokenizer("basic_english")

    data = torch.cat([torch.tensor(vocab(tokenizer(line)), dtype=torch.long) for line in raw_text_iter])
    M = len(data) // seq_len
    r = len(data) % seq_len

    if r == 0:
        data = torch.cat((data, torch.tensor([0], dtype=torch.long)))

    inputs = data[:M * seq_len].reshape(-1, seq_len)
    targets = data[1:M * seq_len + 1].reshape(-1, seq_len)
    return inputs, targets
