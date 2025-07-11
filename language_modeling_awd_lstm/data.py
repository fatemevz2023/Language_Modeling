import os
import torch
from torch.utils.data import Dataset, DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def load_data_iterators(base_path):
    def read_file_gen(filename):
        filepath = os.path.join(base_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            yield from (line.strip() for line in f if line.strip() and not line.startswith('='))
    train_iter = read_file_gen('wiki.train.tokens')
    valid_iter = read_file_gen('wiki.valid.tokens')
    test_iter = read_file_gen('wiki.test.tokens')
    return train_iter, valid_iter, test_iter

tokenizer = get_tokenizer('basic_english')

def build_vocab(train_iter):
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def data_process(raw_text_iter, vocab, seq_len):
    data = torch.cat([torch.LongTensor(vocab(tokenizer(line))) for line in raw_text_iter])
    M = len(data) // seq_len
    r = len(data) % seq_len
    if r != 0:
        data = torch.cat((data, torch.LongTensor([0] * (seq_len - r))))
    inputs = data[:M*seq_len]
    targets = data[1:M*seq_len+1]
    inputs = inputs.reshape(-1, seq_len)
    targets = targets.reshape(-1, seq_len)
    return inputs, targets

class LanguageModelDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def get_dataloaders(base_path, seq_len, batch_size):
    train_iter, valid_iter, test_iter = load_data_iterators(base_path)

    # We need to recreate train_iter for vocab and for data_process since it's a generator
    train_iter_for_vocab, train_iter_for_data = load_data_iterators(base_path)
    vocab = build_vocab(train_iter_for_vocab)

    X_train, y_train = data_process(train_iter_for_data, vocab, seq_len)
    X_valid, y_valid = data_process(valid_iter, vocab, seq_len)
    X_test, y_test = data_process(test_iter, vocab, seq_len)

    train_set = LanguageModelDataset(X_train, y_train)
    valid_set = LanguageModelDataset(X_valid, y_valid)
    test_set = LanguageModelDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, vocab
