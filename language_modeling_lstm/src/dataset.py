import torch
from torch.utils.data import Dataset

class LanguageModelDataset(Dataset):
    """
    A custom dataset class for language modeling.
    Each sample is a pair of input and target sequences.
    """
    def __init__(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Input tensor of shape [num_samples, seq_len]
            targets (Tensor): Target tensor of shape [num_samples, seq_len]
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
