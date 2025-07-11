import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def build_vocab(dataset_iter, specials=['<unk>']):
    """
    Builds vocabulary from tokenized dataset iterator.
    """
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, dataset_iter), specials=specials)
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def save_vocab(vocab, path):
    torch.save(vocab, path)

def load_vocab(path):
    return torch.load(path)
