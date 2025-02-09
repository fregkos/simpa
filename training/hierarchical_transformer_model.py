from tqdm import tqdm

# from preprocessing import preprocess_data
import logging
import os

import numpy as np
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import re
import gc
import glob

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from .paper_dataset import PaperDataset
from .hierarchical_paper_dataset import (
    HierarchicalPaperDataset,
)
from torchmetrics.classification import (
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
    MultilabelPrecisionRecallCurve,
)
import matplotlib.pyplot as plt

import defaults
import random


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        penalty_weight=0.1,
        pos_weight=None,
        device="cuda:0",
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.penalty_weight = penalty_weight
        self.pos_weight = pos_weight if pos_weight else torch.tensor([2.0])
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, logits, targets):
        # Sigmoid activation for probabilities
        logits = logits.to(self.device)
        targets = targets.to(self.device)
        probas = torch.sigmoid(logits).to(self.device)
        # Focal Loss component
        bce_loss_criterion = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=self.pos_weight
        ).to(self.device)
        bce_loss = bce_loss_criterion(logits, targets)
        p_t = targets * probas + (1 - targets) * (1 - probas)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss

        # All-zero penalty component
        all_zero_penalty = self.penalty_weight * torch.mean(
            (probas.sum(dim=1) == 0).float()
        )

        # Combine the losses
        combined_loss = focal_loss.mean() + all_zero_penalty
        return combined_loss


class HierarchicalClassifier:
    def __init__(
        self,
        CSV_PATH: str,
        hyperclass_list: list,  # List of hyperclasses
        label_list: list,  # List of all detailed labels
        hyperclass_to_label_map: dict,  # Dictionary mapping hyperclasses to their possible labels
        epochs: int = 50,
        lr: float = 1e-1,
        batch_size: int = 128,
        threshold: float = 0.9,
        tokenizer_name="bert-base-uncased",
    ):
        self.VEC_DIM = 768
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.hyperclass_names = hyperclass_list
        self.n_hyperclasses = len(hyperclass_list)
        self.n_labels = len(label_list)
        self.hyperclass_to_label_map = hyperclass_to_label_map
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the trunk
        self.trunk = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            safe_serialization=True,
        )
        self.tokenizer_name = tokenizer_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        # Freeze transformer weights
        for param in self.trunk.parameters():
            param.requires_grad = False
        self.trunk.to(device=self.device)

        # Hyperclass classifier head
        self.hyperclass_head = nn.Sequential(
            nn.Linear(self.VEC_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.n_hyperclasses),
        ).to(self.device)

        # Multi-label classifier heads (one for each hyperclass)
        self.label_heads = nn.ModuleDict(
            {
                hyperclass: nn.Sequential(
                    nn.Linear(self.VEC_DIM, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, len(labels)),
                ).to(self.device)
                for hyperclass, labels in hyperclass_to_label_map.items()
            }
        )

        # Calculate metrics
        self.precision_macro = MultilabelPrecision(
            num_labels=self.n_hyperclasses, average="macro"
        ).to(self.device)
        self.recall_macro = MultilabelRecall(
            num_labels=self.n_hyperclasses, average="macro"
        ).to(self.device)
        self.f1_macro = MultilabelF1Score(
            num_labels=self.n_hyperclasses, average="macro"
        ).to(self.device)

        self.precision_micro = MultilabelPrecision(
            num_labels=self.n_hyperclasses, average="micro"
        ).to(self.device)
        self.recall_micro = MultilabelRecall(
            num_labels=self.n_hyperclasses, average="micro"
        ).to(self.device)
        self.f1_micro = MultilabelF1Score(
            num_labels=self.n_hyperclasses, average="micro"
        ).to(self.device)

        self.prc_metric = MultilabelPrecisionRecallCurve(
            num_labels=self.n_hyperclasses
        ).to(self.device)

        # Optimizers
        self.hyperclass_optimizer = torch.optim.Adam(
            self.hyperclass_head.parameters(),
            lr=self.lr,
        )
        self.label_optimizers = {
            hyperclass: torch.optim.Adam(head.parameters(), lr=self.lr)
            for hyperclass, head in self.label_heads.items()
        }

        # Loss functions
        self.hyperclass_criterion = nn.BCEWithLogitsLoss()
        # self.hyperclass_criterion = FocalLoss().to(self.device)
        self.label_criterion = (nn.BCEWithLogitsLoss()).to(
            self.device
        )  # FocalBCELoss(self.n_labels).to(self.device)
        # self.label_criterion = FocalLoss().to(self.device)

        self.threshold = threshold

        # Initialize dataset using HierarchicalPaperDataset
        self._dataset = HierarchicalPaperDataset(
            csv_path=CSV_PATH,
            categorized_labels=hyperclass_to_label_map,
            tokenized_data_path=defaults.HIERARCHICAL_TOKENIZED_DATA_PATH,
        )

    def _train_hyperclass_classifier(self, train_loader, val_loader):
        """Train the hyperclass classifier"""
        self.trunk.eval()
        self.hyperclass_head.train()

        progress_bar = tqdm(range(self.epochs), desc="Training hyperclass classifier")

        for epoch in range(self.epochs):
            total_train_loss = 0
            correct_train = 0
            total_train_samples = 0
            total_val_samples = 0

            for batch in train_loader:  # previously train_loader
                # Get embeddings
                embeddings = batch["embedding"].float().to(self.device)

                hyperclass_labels = batch["hyperclass_labels"].float().to(self.device)

                # Hyperclass prediction
                hyperclass_logits = self.hyperclass_head(embeddings)
                hyperclass_logits /= 10
                loss = self.hyperclass_criterion(hyperclass_logits, hyperclass_labels)

                # Backward pass
                self.hyperclass_optimizer.zero_grad()
                loss.backward()
                self.hyperclass_optimizer.step()

                total_train_loss += loss.item()
                total_train_samples += hyperclass_labels.size(0)

                # Compute accuracy

                pred_hyperclass = torch.sigmoid(hyperclass_logits) > self.threshold

                correct_train += (
                    (pred_hyperclass == hyperclass_labels).float().sum().item()
                )

            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = correct_train / total_train_samples

            total_val_loss = 0
            correct_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    # Get embeddings
                    embeddings = batch["embedding"].float().to(self.device)

                    hyperclass_labels = (
                        batch["hyperclass_labels"].float().to(self.device)
                    )

                    # Hyperclass prediction
                    hyperclass_logits = self.hyperclass_head(embeddings)
                    loss = self.hyperclass_criterion(
                        hyperclass_logits, hyperclass_labels
                    )

                    # Backward pass
                    self.hyperclass_optimizer.zero_grad()
                    # loss.backward()
                    # self.hyperclass_optimizer.step()

                    total_val_loss += loss.item()
                    total_val_samples += hyperclass_labels.size(0)
                    # Compute accuracy
                    pred_hyperclass = torch.sigmoid(hyperclass_logits) > self.threshold

                    correct_val += (
                        (pred_hyperclass == hyperclass_labels).float().sum().item()
                    )

            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = correct_val / total_val_samples
            precision_macro_value = self.precision_macro(
                pred_hyperclass, hyperclass_labels
            )
            recall_macro_value = self.recall_macro(pred_hyperclass, hyperclass_labels)
            f1_macro_value = self.f1_macro(pred_hyperclass, hyperclass_labels)

            precision_micro_value = self.precision_micro(
                pred_hyperclass, hyperclass_labels
            )
            recall_micro_value = self.recall_micro(pred_hyperclass, hyperclass_labels)
            f1_micro_value = self.f1_micro(pred_hyperclass, hyperclass_labels)
            progress_bar.write(
                f"epoch: {epoch+1}/{self.epochs} train loss {avg_train_loss:.4e}, train accuracy: {train_accuracy:.4e} "
                + f"val loss {avg_val_loss:.4e}, val accuracy: {val_accuracy:.4e}"
                + f""" 
Micro=> precision: {precision_micro_value:.2f}, recall: {recall_micro_value:.2f}, f1: {f1_micro_value:.2f}
Macro=> precision: {precision_macro_value:.2f}, recall: {recall_macro_value:.2f}, f1: {f1_macro_value:.2f} """
            )

            progress_bar.update(1)
        # # after the last epoch
        # hyperclass_logits = hyperclass_logits.to("cpu")
        # hyperclass_labels = hyperclass_labels.to("cpu")
        # precisions, recalls, thresholds = self.prc_metric(
        #     hyperclass_logits, hyperclass_labels.to(torch.int)
        # )

        # fig, axes = plt.subplots(
        #     2, 4, figsize=(20, 10)
        # )  # 2 rows, 4 columns for 8 classes

        # for i in range(self.n_hyperclasses):
        #     row, col = divmod(i, 4)  # Determine subplot position
        #     ax = axes[row, col]
        #     ax.plot(recalls[i], precisions[i], label=f"Class {i}")
        #     ax.set_xlabel("Recall")
        #     ax.set_ylabel("Precision")
        #     ax.set_title(f"PRC for Class {i}")
        #     ax.legend()
        #     ax.grid()

        # # Adjust layout and save the figure
        # plt.tight_layout()
        # plt.savefig("plots/prc_curves.png")  # Save as a single file
        # plt.show()

    def _train_label_classifiers(
        self,
        train_loader,
        val_loader,
    ):
        """Train the label classifiers for each hyperclass"""
        self.trunk.eval()

        for hyperclass, head in self.label_heads.items():
            progress_bar = tqdm(
                range(self.epochs), desc=f"Training {hyperclass} classifier"
            )
            head.train()

            for epoch in range(self.epochs):
                total_train_loss = 0
                correct_train = 0
                correct_val = 0
                total_val_loss = 0
                total_train_samples = 0
                total_val_samples = 0

                for batch in train_loader:
                    # Only process samples belonging to current hyperclass
                    hyperclass_indices = [
                        i
                        for i, hc in enumerate(batch["hyperclass"])
                        if hc == hyperclass
                    ]
                    if not hyperclass_indices:
                        continue

                    # Select samples for current hyperclass
                    # docs = [batch["doc"][i] for i in hyperclass_indices]
                    labels = torch.stack(
                        [batch["detailed_labels"][i] for i in hyperclass_indices]
                    ).to(self.device)

                    # Get embeddings
                    embeddings = (
                        torch.stack([batch["embedding"][i] for i in hyperclass_indices])
                        .float()
                        .to(self.device)
                    )
                    # Get relevant labels for this hyperclass
                    hyperclass_labels = labels[
                        :,
                        [
                            self._dataset.label_to_idx[label]
                            for label in self.hyperclass_to_label_map[hyperclass]
                        ],
                    ]

                    # Label prediction
                    label_logits = head(embeddings)
                    loss = self.label_criterion(label_logits, hyperclass_labels)

                    # Backward pass
                    self.label_optimizers[hyperclass].zero_grad()
                    loss.backward()
                    self.label_optimizers[hyperclass].step()

                    total_train_loss += loss.item()
                    total_train_samples += len(hyperclass_indices)

                    # Compute accuracy
                    pred_labels = torch.sigmoid(label_logits) > self.threshold
                    correct_train += (
                        ((pred_labels == 1) & (hyperclass_labels == 1))
                        .float()
                        .sum()
                        .item()
                    )
                with torch.no_grad():
                    for batch in val_loader:
                        # Only process samples belonging to current hyperclass
                        hyperclass_indices = [
                            i
                            for i, hc in enumerate(batch["hyperclass"])
                            if hc == hyperclass
                        ]
                        if not hyperclass_indices:
                            continue

                        # Select samples for current hyperclass
                        # docs = [batch["doc"][i] for i in hyperclass_indices]
                        labels = torch.stack(
                            [batch["detailed_labels"][i] for i in hyperclass_indices]
                        ).to(self.device)

                        # Get embeddings
                        # encoded_input = tokenizer(
                        #     docs,
                        #     truncation=True,
                        #     padding="max_length",
                        #     max_length=512,
                        #     return_tensors="pt",
                        # ).to(self.device)

                        # with torch.no_grad():
                        #     model_output = self.trunk(**encoded_input)
                        #     embeddings = self._mean_pooling(
                        #         model_output, encoded_input["attention_mask"]
                        #     )
                        embeddings = (
                            torch.stack(
                                [batch["embedding"][i] for i in hyperclass_indices]
                            )
                            .float()
                            .to(self.device)
                        )
                        # Get relevant labels for this hyperclass
                        hyperclass_labels = labels[
                            :,
                            [
                                self._dataset.label_to_idx[label]
                                for label in self.hyperclass_to_label_map[hyperclass]
                            ],
                        ]

                        # Label prediction
                        label_logits = head(embeddings)
                        loss = self.label_criterion(label_logits, hyperclass_labels)

                        # Backward pass
                        self.label_optimizers[hyperclass].zero_grad()
                        # loss.backward()
                        # self.label_optimizers[hyperclass].step()

                        total_val_loss += loss.item()
                        total_val_samples += len(hyperclass_indices)

                        # Compute accuracy
                        pred_labels = torch.sigmoid(label_logits) > self.threshold
                        correct_val += (
                            ((pred_labels == 1) & (hyperclass_labels == 1))
                            .float()
                            .sum()
                            .item()
                        )

                if total_train_samples > 0:
                    avg_train_loss = total_train_loss / total_train_samples
                    train_accuracy = correct_train / (
                        total_train_samples
                        * len(self.hyperclass_to_label_map[hyperclass])
                    )

                    progress_bar.write(
                        f"epoch: {epoch + 1}/{self.epochs} train loss {avg_train_loss:.4e}, train accuracy: {train_accuracy:.4e}"
                    )
                if total_val_samples > 0:
                    avg_val_loss = total_val_loss / total_val_samples
                    val_accuracy = correct_val / (
                        total_val_samples
                        * len(self.hyperclass_to_label_map[hyperclass])
                    )

                    progress_bar.write(
                        f"val loss {avg_val_loss:.4e}, val accuracy: {val_accuracy:.4e}"
                    )
                progress_bar.update(1)
            progress_bar.close()

    def train(self):
        """Main training loop"""
        # Split dataset
        np.random.seed(42)
        WANTED_VALUES = 500_000
        random_indices = np.random.choice(
            len(self._dataset), WANTED_VALUES, replace=False
        )
        self.dataset._load_embeddings()  # EXTREMELY IMPORTANT STEP IN PIPELINE!
        subset = Subset(self._dataset, random_indices)
        train_size = int(0.8 * len(subset))
        test_size = len(subset) - train_size
        train_dataset, val_dataset = random_split(subset, [train_size, test_size])

        train_loader = DataLoader(
            # train_subset,  # train_dataset
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            # pin_memory=True,
        )
        val_loader = DataLoader(
            # val_subset,  # val_dataset
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            # pin_memory=True,
        )
        self.val_loader = val_loader
        # Train hyperclass classifier first
        self._train_hyperclass_classifier(
            train_loader,
            val_loader,
        )

        # Then train individual label classifiers
        # self._train_label_classifiers(
        #     train_loader,
        #     val_loader,
        # )

    def _predict(self, text):
        """Predict both hyperclass and labels for a given text"""
        self.trunk.eval()
        self.hyperclass_head.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Encode text
        encoded_input = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # Get embeddings
            model_output = self.trunk(**encoded_input)
            embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

            # Predict hyperclass
            hyperclass_logits = self.hyperclass_head(embeddings)
            pred_hyperclass = torch.sigmoid(hyperclass_logits) > self.threshold

            # For each predicted hyperclass, predict labels
            predictions = {}
            for hyperclass, head in self.label_heads.items():
                if pred_hyperclass[0][
                    list(self.hyperclass_to_label_map.keys()).index(hyperclass)
                ]:
                    label_logits = head(embeddings)
                    pred_labels = torch.sigmoid(label_logits) > self.threshold

                    # Convert predictions to label names
                    hyperclass_label_names = self.hyperclass_to_label_map[hyperclass]
                    pred_label_indices = torch.where(pred_labels[0] == 1)[0]
                    predictions[hyperclass] = [
                        hyperclass_label_names[i] for i in pred_label_indices
                    ]

        return predictions

    def predict(self, text=None):
        """Evaluate the entire validation dataset, compute metrics, and plot PRC curves"""
        self.trunk.eval()
        self.hyperclass_head.eval()

        # Initialize metrics
        total_val_loss = 0
        correct_val = 0
        total_val_samples = 0

        # PRC metric
        self.prc_metric = MultilabelPrecisionRecallCurve(num_labels=self.n_hyperclasses)

        # Metric containers
        all_hyperclass_logits = []
        all_hyperclass_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Get embeddings
                embeddings = batch["embedding"].float().to(self.device)
                hyperclass_labels = batch["hyperclass_labels"].float().to(self.device)

                # Predict hyperclasses
                hyperclass_logits = self.hyperclass_head(embeddings)
                hyperclass_logits /= 10
                loss = self.hyperclass_criterion(hyperclass_logits, hyperclass_labels)

                total_val_loss += loss.item()
                total_val_samples += hyperclass_labels.size(0)

                # Store logits and labels for metrics
                all_hyperclass_logits.append(hyperclass_logits)
                all_hyperclass_labels.append(hyperclass_labels)

                # Accuracy
                pred_hyperclass = (
                    torch.sigmoid(hyperclass_logits) > self.threshold
                ).float()
                correct_val += (pred_hyperclass == hyperclass_labels).sum().item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        val_accuracy = correct_val / total_val_samples

        # Concatenate all logits and labels for metric calculations
        all_hyperclass_logits = torch.cat(all_hyperclass_logits, dim=0)
        all_hyperclass_labels = torch.cat(all_hyperclass_labels, dim=0)

        # Compute metrics
        precision_macro_value = self.precision_macro(
            (all_hyperclass_logits > self.threshold).float(), all_hyperclass_labels
        )
        recall_macro_value = self.recall_macro(
            (all_hyperclass_logits > self.threshold).float(), all_hyperclass_labels
        )
        f1_macro_value = self.f1_macro(
            (all_hyperclass_logits > self.threshold).float(), all_hyperclass_labels
        )
        precision_micro_value = self.precision_micro(
            (all_hyperclass_logits > self.threshold).float(), all_hyperclass_labels
        )
        recall_micro_value = self.recall_micro(
            (all_hyperclass_logits > self.threshold).float(), all_hyperclass_labels
        )
        f1_micro_value = self.f1_micro(
            (all_hyperclass_logits > self.threshold).float(), all_hyperclass_labels
        )

        # Move data to appropriate device
        all_hyperclass_logits = all_hyperclass_logits.cpu()
        all_hyperclass_labels = all_hyperclass_labels.cpu()

        # Get predictions and labels
        true_labels = all_hyperclass_labels.cpu().numpy()
        final_preds = torch.sigmoid(all_hyperclass_logits).cpu().numpy()
        class_names = self.hyperclass_names
        n_classes = len(class_names)  # Number of classes
        # Compute precision-recall and average precision for each class
        precision_recall_curves = []
        average_precisions = []
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(true_labels[:, i], final_preds[:, i])
            ap = average_precision_score(true_labels[:, i], final_preds[:, i])
            precision_recall_curves.append((precision, recall))
            average_precisions.append(ap)
        # Compute micro-average precision-recall curve
        true_labels_flat = true_labels.ravel()
        predicted_probs_flat = final_preds.ravel()
        micro_precision, micro_recall, _ = precision_recall_curve(true_labels_flat, predicted_probs_flat)
        micro_average_precision = average_precision_score(true_labels_flat, predicted_probs_flat)

        # Plot Precision-Recall Curves with Seaborn
        plt.figure(figsize=(10, 8), dpi=600)
        sns.set_theme(style="whitegrid")

        # Plot micro-average curve
        plt.plot(
            micro_recall,
            micro_precision,
            label=f"Micro-average precision-recall (AP = {micro_average_precision:.2f})",
            color="gold",
        )

        # Colors for classes
        palette = sns.color_palette("tab10", n_classes)

        # Plot each class curve
        for i, (precision, recall) in enumerate(precision_recall_curves):
            plt.plot(
                recall,
                precision,
                label=f"{class_names[i]} (AP = {average_precisions[i]:.2f})",
                color=palette[i],
            )

        # Customize the plot
        plt.title("Precision-Recall Curve for Multi-Label Classification", fontsize=26)
        plt.xlabel("Recall", fontsize=24)
        plt.ylabel("Precision", fontsize=24)
        plt.legend(loc="lower left", fontsize='medium', title="Legend")
        plt.tight_layout()
        # plt.savefig("plots/precision_recall_curves.png")  # Save the plot
        plt.show()

        # Print metrics
        print(
            f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}\n"
            f"Macro => Precision: {precision_macro_value:.2f}, Recall: {recall_macro_value:.2f}, F1: {f1_macro_value:.2f}\n"
            f"Micro => Precision: {precision_micro_value:.2f}, Recall: {recall_micro_value:.2f}, F1: {f1_micro_value:.2f}"
        )
        binary_preds = (final_preds >= self.threshold).astype(int)

        # Generate the classification report
        report = classification_report(
            true_labels, binary_preds, target_names=class_names, zero_division=0
        )
        print("Classification report:\n", report)

    def plot_confusion_matrix(self, ):
        """Evaluate the entire validation dataset, compute metrics, and plot confusion matrices"""
        self.trunk.eval()
        self.hyperclass_head.eval()

        # Initialize metrics
        total_val_loss = 0
        correct_val = 0
        total_val_samples = 0

        # Metric containers
        all_hyperclass_logits = []
        all_hyperclass_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Get embeddings
                embeddings = batch["embedding"].float().to(self.device)
                hyperclass_labels = batch["hyperclass_labels"].float().to(self.device)

                # Predict hyperclasses
                hyperclass_logits = self.hyperclass_head(embeddings)
                hyperclass_logits /= 10
                loss = self.hyperclass_criterion(hyperclass_logits, hyperclass_labels)

                total_val_loss += loss.item()
                total_val_samples += hyperclass_labels.size(0)

                # Store logits and labels for metrics
                all_hyperclass_logits.append(hyperclass_logits)
                all_hyperclass_labels.append(hyperclass_labels)

                # Accuracy
                pred_hyperclass = (
                    torch.sigmoid(hyperclass_logits) > self.threshold
                ).float()
                correct_val += (pred_hyperclass == hyperclass_labels).sum().item()

        # Concatenate all logits and labels for metric calculations
        all_hyperclass_logits = torch.cat(all_hyperclass_logits, dim=0)
        all_hyperclass_labels = torch.cat(all_hyperclass_labels, dim=0)

        # # Move data to appropriate device
        all_hyperclass_logits = all_hyperclass_logits.cpu()
        all_hyperclass_labels = all_hyperclass_labels.cpu()

        # Get predictions and labels
        true_labels = all_hyperclass_labels.cpu().numpy()
        final_preds = (
            torch.sigmoid(all_hyperclass_logits).numpy() > self.threshold
        ).astype(int)
        class_names = self.hyperclass_names
        class_names = [ # overwrite for simplicity in plot
            "cs",
            "econ",
            "eess",
            "math",
            "physics",
            "q-bio",
            "q-fin",
            "stat",
        ]
        n_classes = len(class_names)  # Number of classes

        # Compute and plot confusion matrices
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), dpi=600)
        axes = axes.ravel()

        for i in range(n_classes):
            cm = confusion_matrix(true_labels[:, i], final_preds[:, i])
            disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
            disp.plot(ax=axes[i], values_format=".4g")
            disp.ax_.set_title(f"Class: {class_names[i]}")

            if i < 4:
                disp.ax_.set_xlabel("")
            if i % 4 != 0:
                disp.ax_.set_ylabel("")

            disp.im_.colorbar.remove()

        plt.subplots_adjust(wspace=0.15, hspace=0.1)
        fig.colorbar(disp.im_, ax=axes)
        plt.savefig("plots/confusion_matrices.png")
        plt.show()

    def extract_and_save_embeddings(
        self,
        HIERARCHICAL_EMBEDDINGS_DATA_PATH=defaults.HIERARCHICAL_EMBEDDINGS_DATA_PATH,
    ):
        @torch.jit.script
        def mean_pooling(
            token_embeddings: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return F.normalize(embeddings, p=2.0, dim=1)

        dataset_loader = DataLoader(
            self._dataset,
            self.batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 4),  # Optimize number of workers
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between batches
            prefetch_factor=2,  # Prefetch next batches
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trunk.to(device)
        self.trunk.eval()

        # Precalculate total size
        total_samples = len(
            self._dataset
        )  # More efficient than iterating through loader
        sample_embedding_size = self.trunk.config.hidden_size
        all_embeddings = torch.zeros(
            (total_samples, sample_embedding_size),
            pin_memory=True if torch.cuda.is_available() else False,
        )

        print("Extracting embeddings from papers...")

        progress_bar = tqdm(
            total=len(dataset_loader),
            desc="Extracting embeddings",
            unit=f"{self.batch_size} embeddings",
        )

        start_idx = 0

        # Pre-allocate tensors for tokenizer output
        max_length = 512
        tokenizer_kwargs = {
            "truncation": True,
            "padding": "max_length",
            "max_length": max_length,
            "return_tensors": "pt",
        }

        with torch.no_grad():
            for batch in dataset_loader:
                docs = batch["doc"]
                batch_size = len(docs)

                # Batch tokenization
                encoded_input = self.tokenizer(docs, **tokenizer_kwargs).to(
                    device, non_blocking=True
                )  # Non-blocking transfer

                # Forward pass with mixed precision
                model_output = self.trunk(**encoded_input)
                embeddings = mean_pooling(
                    model_output[0], encoded_input["attention_mask"]
                )

                # Efficient CPU transfer and storage
                end_idx = start_idx + batch_size
                all_embeddings[start_idx:end_idx].copy_(
                    embeddings.cpu(), non_blocking=True
                )
                start_idx = end_idx

                progress_bar.update(1)

        progress_bar.close()

        # Save embeddings efficiently
        save_path = HIERARCHICAL_EMBEDDINGS_DATA_PATH
        torch.save(all_embeddings, save_path, _use_new_zipfile_serialization=True)
        print(f"Saved embeddings tensors to {save_path}")
        # Clear CUDA cache
        torch.cuda.empty_cache()
        # Clear DataLoader workers
        dataset_loader._iterator = None
        # Force garbage collection
        import gc

        gc.collect()

        return all_embeddings

    @property
    def dataset(self) -> Dataset:
        return self._dataset
