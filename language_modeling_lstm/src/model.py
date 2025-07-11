import torch
import torch.nn as nn

class LanguageModel(nn.Module):
    """
    LSTM-based Language Model consisting of:
    - Embedding layer
    - LSTM layer(s)
    - Fully connected output layer
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 dropout_embd=0.5, dropout_rnn=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_embd)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            dropout=dropout_rnn, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Optional weight initialization
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, src):
        x = self.dropout(self.embedding(src))
        output, _ = self.lstm(x)
        return self.fc(output)
