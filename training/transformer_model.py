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


class FocalBCELoss(nn.Module):
    def __init__(self, num_labels=155):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.ones(num_labels) * 10  # Start with high positive weight
        )

    def forward(self, outputs, targets):
        # BCE loss
        bce_loss = self.bce(outputs, targets)

        # Add penalty for predicting all zeros
        prob_outputs = torch.sigmoid(outputs)
        all_zeros_penalty = torch.mean((1 - prob_outputs).pow(2))

        return bce_loss + 0.1 * all_zeros_penalty


# criterion = FocalBCELoss()


class TransformerClassifier:
    def __init__(
        self,
        CSV_PATH: str,
        label_list: list,  # Number of labels for multi-label classification
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        n_neurons: int = 128,
        threshold: float = 0.9,
        tokenizer_name="bert-base-uncased",
    ):
        self.VEC_DIM = 768
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_neurons = n_neurons
        self.n_labels = len(label_list)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trunk = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            safe_serialization=True,
        )
        self.tokenizer_name = tokenizer_name
        # freeze transformer weights
        for param in self.trunk.parameters():
            param.requires_grad = False
        self.trunk.to(device=self.device)

        # Define the head for multi-label classification
        self.head = nn.Sequential( # simple head
            nn.Linear(self.VEC_DIM, self.n_labels)
            # No Sigmoid here, we use BCEWithLogitsLoss which combines sigmoid
        ).to(self.device)

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.head.parameters(),
            lr=self.lr,
        )
        self.dataset = PaperDataset(
            CSV_PATH, label_list, tokenized_data_path=defaults.TOKENIZED_DATA_PATH
        )
        # self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        # self.criterion = nn.BCELoss()
        self.criterion = (
            nn.BCEWithLogitsLoss()
        )  # Use BCEWithLogitsLoss for multi-label classification
        self.criterion = FocalBCELoss(len(label_list)).to(self.device)
        self.threshold = threshold

    def train_or_load_transformer_model(self):
        pass

    def _train_transformer_model(
        self,
    ):
        # Split into train and test datasets
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(
            self.dataset, [train_size, test_size]
        )

        # Create DataLoaders for training and validation
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Store metrics
        training_metrics = {"loss": [], "accuracy": []}
        validation_metrics = {"loss": [], "accuracy": []}
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, cache_dir=None)

        progress_bar_epoch = tqdm(
            total=self.epochs,
            desc="Training epoch",
            unit="epoch",
        )
        for epoch in range(self.epochs):
            # Training phase
            self.trunk.eval()  # let this be .eval()! and disable any dropout etc layers!
            self.head.train()

            total_train_loss = 0
            correct_train = 0
            total_train_samples = 0
            for batch in train_loader:
                # input_ids = batch["input_ids"].to(self.device)
                # attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].float().to(self.device)

                # encoded_input = batch['doc']

                encoded_input = tokenizer(
                    batch["doc"],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                # Forward pass
                with torch.no_grad():  # No gradients for the transformer trunk
                    # model_output = self.trunk(input_ids=input_ids, attention_mask=attention_mask)
                    model_output = self.trunk(**encoded_input)
                    # [128, 253, 768] -> [128, 1, 768] -> [128, 768]
                    # [128 true labels] =~= [128 predicted labels]
                    embeddings = self._mean_pooling(
                        model_output, encoded_input["attention_mask"]
                    )
                    

                logits = self.head(embeddings)  # 768 -> 155

                # Compute loss
                loss = self.criterion(logits, labels)
                total_train_loss += loss.item()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Compute accuracy (multi-label)
                pred_labels = torch.sigmoid(
                    logits
                )  # Apply sigmoid to get probabilities
                pred_labels = (
                    pred_labels > self.threshold
                ).float()  # Use threshold to classify as 0 or 1
                # progress_bar_epoch.write(f'predicted: {pred_labels.sum(1)} \n real: {labels.sum(1)} \n predictions: {((pred_labels == 1) & (labels == 1)).sum(1)}')
                # progress_bar_epoch.write(f'predictions: {((pred_labels == 1) & (labels == 1)).sum(1)}')
                # This will count all positions where both tensors have 1s
                correct_train += (
                    ((pred_labels == 1) & (labels == 1)).float().sum().item()
                )

                # If you also want to track precision/recall, you might want:
                # true_positives = ((pred_labels == 1) & (labels == 1)).float().sum().item()
                # false_positives = ((pred_labels == 1) & (labels == 0)).float().sum().item()
                # false_negatives = ((pred_labels == 0) & (labels == 1)).float().sum().item()
                # 128 * 155 = 19840
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
                    # input_ids = batch["input_ids"].to(self.device)
                    # attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].float().to(self.device)

                    encoded_input = tokenizer(
                        batch["doc"],
                        truncation=True,
                        padding="max_length",
                        max_length=512,
                        return_tensors="pt",
                    ).to(self.device)

                    # Forward pass
                    # model_output = self.trunk(input_ids=input_ids, attention_mask=attention_mask)
                    model_output = self.trunk(**encoded_input)
                    embeddings = self._mean_pooling(
                        model_output, encoded_input["attention_mask"]
                    )
                    
                    logits = self.head(embeddings)

                    # Compute loss
                    loss = self.criterion(logits, labels)
                    total_val_loss += loss.item()

                    # Compute accuracy (multi-label)
                    pred_labels = torch.sigmoid(
                        logits
                    )  # Apply sigmoid to get probabilities
                    pred_labels = (
                        pred_labels > self.threshold
                    ).float()  # Use threshold to classify as 0 or 1

                    correct_val += (
                        ((pred_labels == 1) & (labels == 1)).float().sum().item()
                    )
                    total_val_samples += labels.numel()

            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = correct_val / total_val_samples

            validation_metrics["loss"].append(avg_val_loss)
            validation_metrics["accuracy"].append(val_accuracy)

            # Print metrics for the current epoch, keeping the progress bar
            # at the bottom of the terminal using .write(str)
            progress_bar_epoch.write(
                f"Epoch {epoch + 1}/{self.epochs}, "
                f"Train Loss: {avg_train_loss:.8e}, Train Accuracy: {train_accuracy:.8e}, "
                f"Val Loss: {avg_val_loss:.8e}, Val Accuracy: {val_accuracy:.8e}"
            )
            progress_bar_epoch.update(1)
        progress_bar_epoch.close()
        # Return the metrics
        return training_metrics, validation_metrics

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        # print("token_embeddings size: ", token_embeddings.size())
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

