import torch
import tqdm
import torch.nn.utils
from .utils import AverageMeter

def train_one_epoch(model, dataloader, loss_fn, optimizer, metric, device, epoch=None, clip=0.25):
    model.train()
    loss_meter = AverageMeter()
    metric.reset()

    with tqdm.tqdm(dataloader, unit='batch') as tepoch:
        for inputs, targets in tepoch:
            if epoch is not None:
                tepoch.set_description(f"Epoch {epoch}")

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

            loss_meter.update(loss.item(), n=inputs.size(0))
            metric.update(outputs, targets)

            tepoch.set_postfix(loss=loss_meter.avg, perplexity=metric.compute().item())

    return model, loss_meter.avg, metric.compute().item()

def evaluate(model, dataloader, loss_fn, metric, device):
    model.eval()
    loss_meter = AverageMeter()
    metric.reset()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())

            loss_meter.update(loss.item(), n=inputs.size(0))
            metric.update(outputs, targets)

    return loss_meter.avg, metric.compute().item()
