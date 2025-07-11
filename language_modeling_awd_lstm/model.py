import torch
from torch import nn
import torch.nn.functional as F

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def _setup(self):
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = lambda: None
            for name_w in self.weights:
                w = getattr(self.module, name_w)
                del self.module._parameters[name_w]
                self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            mask = F.dropout(torch.ones_like(raw_w), p=self.dropout, training=self.training) * (1 - self.dropout)
            setattr(self.module, name_w, raw_w * mask)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new_empty((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
    padding_idx = embed.padding_idx if embed.padding_idx is not None else -1
    return F.embedding(words, masked_embed_weight, padding_idx, embed.max_norm,
                       embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

class LockedDropout(nn.Module):
    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class AWD_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 dropoute=0.1, dropouti=0.65, dropouth=0.3, dropouto=0.4,
                 weight_drop=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.lockdrop = LockedDropout()
        self.dropoute = dropoute
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropouto = dropouto

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstms = nn.ModuleList()
        self.lstms.append(nn.LSTM(embedding_dim, hidden_dim, 1))
        self.lstms.append(nn.LSTM(hidden_dim, hidden_dim, 1))
        self.lstms.append(nn.LSTM(hidden_dim, embedding_dim, 1))

        if weight_drop > 0:
            self.lstms = nn.ModuleList([WeightDrop(lstm, ['weight_hh_l0'], dropout=weight_drop) for lstm in self.lstms])

        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.fc.weight = self.embedding.weight

    def forward(self, x):
        emb = embedded_dropout(self.embedding, x, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        output = emb
        for i, lstm in enumerate(self.lstms):
            output, _ = lstm(output)
            if i != self.num_layers - 1:
                output = self.lockdrop(output, self.dropouth)

        output = self.lockdrop(output, self.dropouto)
        output = self.fc(output)
        return output
