import torch
import torch.nn as nn
import torchmetrics as tm
from data import get_dataloaders
from model import AWD_LSTM
from config import *
from utils import set_seed

device = device

def evaluate(model, data_loader, loss_fn, metric):
    model.eval()
    loss_meter = 0
    metric.reset()
    total_loss = 0
    total_samples = 0
    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = inputs.t().to(device)
            targets = targets.t().to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())
            total_loss += loss.item() * targets.size(1)
            total_samples += targets.size(1)
            metric.update(outputs, targets)
    avg_loss = total_loss / total_samples
    return avg_loss, metric.compute().item()

if __name__ == "__main__":
    set_seed(seed)
    train_loader, valid_loader, test_loader, vocab = get_dataloaders('data/wikitext-2', seq_len, batch_size)
    model = torch.load('model.pt').to(device)
    loss_fn = nn.CrossEntropyLoss()
    metric = tm.text.Perplexity().to(device)

    valid_loss, valid_ppl = evaluate(model, valid_loader, loss_fn, metric)
    print(f'Validation Loss: {valid_loss:.4f}, Perplexity: {valid_ppl:.4f}')

    test_loss, test_ppl = evaluate(model, test_loader, loss_fn, metric)
    print(f'Test Loss: {test_loss:.4f}, Perplexity: {test_ppl:.4f}')
