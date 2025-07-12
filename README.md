# 📚Language Modeling with WikiText-2

This repository contains an implementation of an AWD-LSTM-based language model trained on the [WikiText-2](https://huggingface.co/datasets/wikitext/viewer/wikitext-2) dataset using PyTorch.

## 🧠 Project Overview

The goal of this project is to build and train a neural language model that can learn the structure of English text and generate fluent sentences.

We initially trained a basic LSTM-based model on the WikiText-2 dataset and achieved a **perplexity of around 130**, which is a reasonable baseline.

Later, inspired by the techniques introduced in the paper:

> **"Regularizing and Optimizing LSTM Language Models"**
> Stephen Merity, Nitish Shirish Keskar, and Richard Socher
> [https://arxiv.org/abs/1708.02182](https://arxiv.org/abs/1708.02182)

we improved the architecture by applying:

* **Dropout Variants** (input, output, hidden, embedding)
* **WeightDrop** on LSTM recurrent weights
* **LockedDropout**
* **Tied Weights** between the input embedding and output projection layers

These enhancements significantly improved the generalization capability of the model and reduced the **perplexity to under 90**, which aligns with results reported in the paper.

---

## 📁 Project Structure

```
project-root/
│
├── src/
│   ├── dataset.py           # Loading and preprocessing WikiText-2
│   ├── model.py             # AWD-LSTM model implementation
│   ├── train.py             # Training and validation routines
│   ├── generate.py          # Text generation from a prompt
│   └── utils.py             # Helper classes and functions
│
├── data/                    # (Optional) Local dataset storage
├── notebooks/               # Colab notebooks (optional)
├── model.pt                 # Best model checkpoint
├── vocab.pt                # Saved vocabulary
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🧪 Results

| Model Variant   | Perplexity (Valid) | Perplexity (Test) |
| --------------- | ------------------ | ----------------- |
| Vanilla LSTM    | ≈ 130              | ≈ 132             |
| AWD-LSTM (ours) | **< 90**           | **< 92**          |

---

## 🔥 Example Generation

Given the prompt:

```
"Once upon a time, there lived a young princess named"
```

The model may generate something like:

> *"Once upon a time, there lived a young princess named Elizabeth who dreamed of exploring the kingdom beyond the castle walls."*

---

## 🛠 Dependencies

To install required packages:

```bash
pip install -r requirements.txt
```

Tested with:

* `torch==2.2.2`
* `torchtext==0.17.2`
* `torchmetrics==1.3.1`
* `numpy==2.0.0`
* `tqdm==4.67.1`

---

## 📊 Optional: Weights & Biases Logging

You can monitor training via [Weights & Biases](https://wandb.ai/) by enabling `wandb_enable = True` in the config section and placing your API key in a local `key` file.

---

## 📌 References

* [AWD-LSTM Paper (Merity et al., 2017)](https://arxiv.org/abs/1708.02182)
* [WikiText-2 Dataset](https://paperswithcode.com/dataset/wikitext-2)

---


