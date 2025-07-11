# main.py

import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from src.vocab import build_vocab, save_vocab, load_vocab
from src.dataset import LanguageModelDataset
from src.model import LanguageModel
from src.train import train_one_epoch, evaluate
from src.utils import data_process
from src.utils import AverageMeter
from torchtext.data.utils import get_tokenizer

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_dir = './wikitext-2'
vocab_path = './vocab.pt'
seq_len = 35
batch_size = 20
embedding_dim = 300
hidden_dim = 512
num_layers = 2
dropout_embd = 0.5
dropout_rnn = 0.2
lr = 0.05
wd = 1e-6
num_epochs = 5

# 1. Load dataset
def load_data_iterators(base_path):
    def read_file_gen(filename):
        with open(os.path.join(base_path, filename), 'r', encoding='utf-8') as f:
            yield from (line.strip() for line in f if line.strip() and not line.startswith('='))
    return read_file_gen('wiki.train.tokens'), read_file_gen('wiki.valid.tokens'), read_file_gen('wiki.test.tokens')

train_iter, valid_iter, test_iter = load_data_iterators(data_dir)

# 2. Build or load vocabulary
if os.path.exists(vocab_path):
    vocab = load_vocab(vocab_path)
    print("Vocabulary loaded.")
else:
    vocab = build_vocab(train_iter)
    save_vocab(vocab, vocab_path)
    print("Vocabulary built and saved.")

# 3. Re-initialize iterators
train_iter, valid_iter, test_iter = load_data_iterators(data_dir)

# 4. Convert text to tensors
X_train, y_train = data_process(train_iter, vocab, seq_len)
X_valid, y_valid = data_process(valid_iter, vocab, seq_len)
X_test,  y_test  = data_process(test_iter,  vocab, seq_len)

# 5. Create datasets and dataloaders
train_set = LanguageModelDataset(X_train, y_train)
valid_set = LanguageModelDataset(X_valid, y_valid)
test_set  = LanguageModelDataset(X_test, y_test)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)
test_loader  = DataLoader(test_set, batch_size=batch_size)

# 6. Create model
model = LanguageModel(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout_embd=dropout_embd,
    dropout_rnn=dropout_rnn
).to(device)

# 7. Optimizer, loss, metric
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
loss_fn = nn.CrossEntropyLoss()
metric = torchmetrics.text.Perplexity().to(device)

# 8. Training loop
best_valid_loss = float('inf')
for epoch in range(1, num_epochs + 1):
    print(f"\n🔁 Epoch {epoch}")
    model, train_loss, train_ppl = train_one_epoch(model, train_loader, loss_fn, optimizer, metric, device, epoch)
    val_loss, val_ppl = evaluate(model, valid_loader, loss_fn, metric, device)

    print(f"✅ Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.4f}")

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model, 'model.pt')
        print("💾 Model saved!")

# 9. Final test
model = torch.load('model.pt')
test_loss, test_ppl = evaluate(model, test_loader, loss_fn, metric, device)
print(f"\n🎯 Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.4f}")
