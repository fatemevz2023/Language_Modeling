import torch
from data import get_dataloaders
from model import AWD_LSTM
from config import *
from utils import set_seed
import argparse

device = device

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    indices = vocab(tokenizer(prompt))
    itos = vocab.get_itos()

    for _ in range(max_seq_len):
        src = torch.LongTensor(indices).to(device)
        with torch.no_grad():
            prediction = model(src)
        probs = torch.softmax(prediction[-1]/temperature, dim=0)
        idx = vocab['<unk>']
        while idx == vocab['<unk>']:
            idx = torch.multinomial(probs, num_samples=1).item()
        token = itos[idx]
        prompt += ' ' + token
        if idx == vocab['.']:
            return prompt
        indices.append(idx)
    return prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='Once upon a time,')
    parser.add_argument('--max_len', type=int, default=35)
    parser.add_argument('--temp', type=float, default=0.5)
    args = parser.parse_args()

    set_seed(seed)
    _, _, _, vocab = get_dataloaders('data/wikitext-2', seq_len, batch_size)
    model = torch.load('model.pt').to(device)
    model.eval()

    generated_text = generate(args.prompt, args.max_len, args.temp, model, get_tokenizer('basic_english'), vocab, seed)
    print(generated_text)
