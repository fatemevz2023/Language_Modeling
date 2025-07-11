import torch
from torch import optim
import torch.nn as nn
import tqdm

from utils import AverageMeter, set_seed
from data import get_dataloaders
from model import AWD_LSTM
from config import *

device = device

def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=None):
    model.train()
    loss_meter = AverageMeter()
    metric.reset()
    with tqdm.tqdm(train_loader, unit='batch') as tepoch:
        for inputs, targets in tepoch:
            if epoch:
                tepoch.set_description(f'Epoch {epoch}')
            inputs = inputs.t().to(device)
            targets = targets.t().to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())
            loss.backward()
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
            optimizer.zero_grad()
            loss_meter.update(loss.item(), n=targets.size(1))
            metric.update(outputs, targets)
            tepoch.set_postfix(loss=loss_meter.avg, perplexity=metric.compute().item())
    return model, loss_meter.avg, metric.compute().item()

if __name__ == '__main__':
    set_seed(seed)
    train_loader, valid_loader, test_loader, vocab = get_dataloaders('data/wikitext-2', seq_len, batch_size)

    model = AWD_LSTM(len(vocab), embedding_dim, hidden_dim, num_layers,
                     dropoute, dropouti, dropouth, dropouto,
                     weight_drop).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    import torchmetrics as tm
    metric = tm.text.Perplexity().to(device)

    for epoch in range(1, num_epochs+1):
        model, train_loss, train_ppl = train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch)
        # You can add validation here if you want
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Perplexity = {train_ppl:.4f}")

    torch.save(model, 'model.pt')
