import argparse
import torch
from model import AWD_LSTM
from config import *
from data import get_dataloaders, tokenizer
from train import train_one_epoch
from evaluate import evaluate
from generate import generate
from utils import set_seed
import torch.nn as nn
import torchmetrics as tm
import torch.optim as optim

def main():
    # Set up argument parser to handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate", "generate"], help="Choose action to perform")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for text generation")
    parser.add_argument("--max_len", type=int, default=35, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    # Set seed for reproducibility and prepare data loaders
    set_seed(seed)
    device_ = device
    train_loader, valid_loader, test_loader, vocab = get_dataloaders('data/wikitext-2', seq_len, batch_size)
    loss_fn = nn.CrossEntropyLoss()
    metric = tm.text.Perplexity().to(device_)

    if args.mode == "train":
        # Initialize model and optimizer for training
        model = AWD_LSTM(len(vocab), embedding_dim, hidden_dim, num_layers,
                         dropoute, dropouti, dropouth, dropouto, weight_drop).to(device_)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

        print("Training started...")
        # Train the model over multiple epochs
        for epoch in range(1, num_epochs + 1):
            model, _, ppl = train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch)
            print(f"Epoch {epoch}: Train Perplexity = {ppl:.2f}")
        # Save the trained model
        torch.save(model, "model.pt")
        print("Training completed. Model saved to model.pt")

    elif args.mode == "evaluate":
        # Load the trained model and evaluate on validation and test sets
        model = torch.load("model.pt").to(device_)
        val_loss, val_ppl = evaluate(model, valid_loader, loss_fn, metric)
        test_loss, test_ppl = evaluate(model, test_loader, loss_fn, metric)
        print(f"Validation Perplexity: {val_ppl:.2f}")
        print(f"Test Perplexity: {test_ppl:.2f}")

    elif args.mode == "generate":
        # Load the trained model and generate text from a prompt
        model = torch.load("model.pt").to(device_).eval()
        text = generate(args.prompt, args.max_len, args.temperature, model, tokenizer, vocab)
        print("Generated Text:")
        print(text)

if __name__ == "__main__":
    main()
