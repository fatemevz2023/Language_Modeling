# Language Modeling with LSTM

This project implements a language model based on an LSTM neural network to predict the next tokens in text sequences. The goal is to train a deep learning model that can generate new text based on given input data.

## Features

- Text preprocessing using the torchtext library
- LSTM model implementation with PyTorch
- Training on the WikiText-2 dataset
- Evaluation using Perplexity metric
- Text generation based on a given prompt

## Project Structure

- `data.py`: Data loading and preprocessing
- `model.py`: Definition of the LSTM model
- `train.py`: Model training script
- `evaluate.py`: Model evaluation script
- `generate.py`: Text generation using the trained model
- `utils.py`: Helper functions like setting seeds and averaging
- `config.py`: Project configurations and hyperparameters
- `main.py`: Main script to run training, evaluation, or generation modes

## Requirements

- Python 3.7+
- PyTorch
- torchtext
- numpy
- torchmetrics
- tqdm

You can install the dependencies via the requirements file:

```bash
pip install -r requirements.txt
