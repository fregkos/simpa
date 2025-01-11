from tqdm import tqdm

# from preprocessing import preprocess_data
import logging
import os


from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import re
import gc
import glob

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from .paper_dataset import PaperDataset

import defaults


class TransformerClassifier:
    def __init__(
        self,
        dataset: PaperDataset,
        create_new_model: bool = False,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        n_neurons: int = 256,
        labels: int = 155,  # IMPORTANT!
    ):
        self.dataset = dataset  # the PaperDataset
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_neurons = n_neurons
        self.labels = labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trunk = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            safe_serialization=True,
        )
        self.head = 

    def train_or_load_transformer_model(
        self,
    ):
        pass

    def _train_transformer_model(
        self,
    ):
        pass

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        print("token_embeddings size: ", token_embeddings.size())
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


# In order to classify a document based on the sentence transformer
# we need to prepend the 'classification: ' text before each piece of text ?
# based on https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    print("token_embeddings size: ", token_embeddings.size())
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def train_transformer_model(csv_path: str):
    sentences = ["classification: the quick brown fox jumped over the fence"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        safe_serialization=True,
    )

    model = model.to(device)
    model.eval()
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    # move to appropriate device
    encoded_input = {
        key: value.to(device) for key, value in encoded_input.items()
    }  # Move tensors to GPU

    print(f"{encoded_input=}")
    with torch.no_grad():
        model_output = model(**encoded_input)

    attention_mask = encoded_input["attention_mask"]
    embeddings = mean_pooling(model_output, attention_mask)

    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu()
    print("final embedding size: ", embeddings.size())
