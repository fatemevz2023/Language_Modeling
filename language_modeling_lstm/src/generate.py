import torch
import torch.nn.functional as F

def generate(prompt, max_len, temperature, model, tokenizer, vocab, device='cpu', seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    indices = vocab(tokenizer(prompt))
    itos = vocab.get_itos()

    for _ in range(max_len):
        src = torch.LongTensor(indices).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(src)

        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()

        if idx == vocab['<unk>']:
            continue

        token = itos[idx]
        prompt += ' ' + token

        if token == '.':
            break

        indices.append(idx)

    return prompt
