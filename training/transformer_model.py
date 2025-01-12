from tqdm import tqdm

# from preprocessing import preprocess_data
import logging
import os


from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


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
        n_labels: int = 155,  # Number of labels for multi-label classification
        threshold: float = 0.9,
    ):
        self.VEC_DIM = 768
        self.dataset = dataset  # the PaperDataset
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_neurons = n_neurons
        self.n_labels = n_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trunk = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            safe_serialization=True,
        ).to(self.device)

        for param in self.trunk.parameters():
            param.requires_grad = False

        # Define the head for multi-label classification
        self.head = nn.Sequential(
            nn.Linear(self.VEC_DIM, self.n_neurons),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.n_neurons, self.n_labels),
            # No Sigmoid here, we use BCEWithLogitsLoss which combines sigmoid
        ).to(self.device)

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(
            list(self.trunk.parameters()) + list(self.head.parameters()), lr=self.lr,
        )
        self.criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.threshold = threshold
    def train_or_load_transformer_model(self):
        pass

    def _train_transformer_model(self, ):
        # Split into train and test datasets
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])

        # Create DataLoaders for training and validation
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True
        )
        val_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, pin_memory=True
        )

        # Store metrics
        training_metrics = {"loss": [], "accuracy": []}
        validation_metrics = {"loss": [], "accuracy": []}

        for epoch in range(self.epochs):
            # Training phase
            self.trunk.train()
            self.head.train()

            total_train_loss = 0
            correct_train = 0
            total_train_samples = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].float().to(self.device)

                # Forward pass
                with torch.no_grad():  # No gradients for the transformer trunk
                    model_output = self.trunk(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = self._mean_pooling(model_output, attention_mask)

                logits = self.head(embeddings)

                # Compute loss
                loss = self.criterion(logits, labels)
                total_train_loss += loss.item()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Compute accuracy (multi-label)
                pred_labels = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
                pred_labels = (pred_labels > self.threshold).float()  # Use threshold to classify as 0 or 1

                correct_train += (pred_labels == labels).float().sum().item()
                total_train_samples += labels.numel()

            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = correct_train / total_train_samples

            training_metrics["loss"].append(avg_train_loss)
            training_metrics["accuracy"].append(train_accuracy)

            # Validation phase
            self.trunk.eval()
            self.head.eval()

            total_val_loss = 0
            correct_val = 0
            total_val_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].float().to(self.device)

                    # Forward pass
                    model_output = self.trunk(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = self._mean_pooling(model_output, attention_mask)
                    logits = self.head(embeddings)

                    # Compute loss
                    loss = self.criterion(logits, labels)
                    total_val_loss += loss.item()

                    # Compute accuracy (multi-label)
                    pred_labels = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
                    pred_labels = (pred_labels > 0.9).float()  # Use threshold to classify as 0 or 1

                    correct_val += (pred_labels == labels).float().sum().item()
                    total_val_samples += labels.numel()

            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = correct_val / total_val_samples

            validation_metrics["loss"].append(avg_val_loss)
            validation_metrics["accuracy"].append(val_accuracy)

            # Print metrics for the current epoch
            print(
                f"Epoch {epoch + 1}/{self.epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

        # Return the metrics
        return training_metrics, validation_metrics

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        # print("token_embeddings size: ", token_embeddings.size())
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
    # print("token_embeddings size: ", token_embeddings.size())
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
