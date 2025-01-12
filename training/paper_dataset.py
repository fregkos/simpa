import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoTokenizer
import os
import pandas as pd
from tqdm import tqdm


class _PaperDataset(Dataset):
    """Obsolete code, soon to be REMOVED"""

    def __init__(
        self,
        csv_path,
        label_list,
        tokenizer_name="bert-base-uncased",
        max_length=512,
    ):
        """
        Args:
            csv_path (str): Path to the CSV file.
            tokenizer_name (str): Name or path of the pretrained tokenizer.
            label_list (list): List of all possible labels (tags).
            max_length (int): Maximum sequence length for tokenization.
        """
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        self.length = len(self.df)
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Convert tags to numeric labels
        self.label_to_idx = {label: idx for idx, label in enumerate(label_list)}

        # Preallocate memory for tokenized inputs and labels
        num_samples = len(self.df)
        # TODO look into max_length. BERT usees 512 !!! is this an issue?
        self.input_ids = torch.zeros((num_samples, max_length), dtype=torch.long)
        self.attention_mask = torch.zeros((num_samples, max_length), dtype=torch.long)
        self.labels = torch.zeros((num_samples, len(label_list)), dtype=torch.float)

        # Tokenize and encode the data
        for index, row in self.df.iterrows():
            # Tokenize the document
            tokens = self.tokenizer(
                row["document"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            # Store tokenized input
            self.input_ids[index] = tokens["input_ids"].squeeze()
            self.attention_mask[index] = tokens["attention_mask"].squeeze()

            # Process tags and store labels
            tags = row["tags"].split(" ")  # Assuming tags are comma-separated
            for tag in tags:
                if tag in self.label_to_idx:
                    self.labels[index][
                        self.label_to_idx[tag]
                    ] = 1.0  # Set corresponding label to 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def prepare_dataset(csv_path, 
                    tokenizer_name, 
                    cache_dir, 
                    max_length, 
                    label_list, 
                    tokenized_data_path, 
                    limit):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    # Load CSV data
    df = pd.read_csv(csv_path)
    num_samples = len(df)

    # Preallocate tensors for tokenized data
    input_ids = torch.zeros((num_samples, max_length), dtype=torch.long)
    attention_mask = torch.zeros((num_samples, max_length), dtype=torch.long)
    labels = torch.zeros((num_samples, len(label_list)), dtype=torch.float)

    # Tokenize documents and encode labels
    label_list = label_list
    label_list_length = len(label_list)
    label_to_idx = {label: idx for idx, label in enumerate(label_list)}
    # TODO look into max_length. BERT usees 512 !!! is this an issue?
    progress_bar = tqdm(
        total=len(df) if not limit else limit,
        desc="Extracting tensors from papers & assigning tags",
        unit="papers",
    )
    for index, row in df.iterrows():
        if index > limit:
            break
        # Tokenize the document
        tokens = tokenizer(
            row["doc"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        # these next two lines might cause issues due to squeezing
        # and dimensionalites
        input_ids[index] = tokens["input_ids"].squeeze()
        attention_mask[index] = tokens["attention_mask"].squeeze()

        # Process tags and store labels
        tags = row["categories"].split(" ")  # Assuming tags are space separated
        for tag in tags:
            if tag in label_to_idx:
                labels[index][label_to_idx[tag]] = 1.0

        progress_bar.update(1)
    progress_bar.close()

    # Save the tokenized dataset to disk
    tokenized_data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    torch.save(tokenized_data, tokenized_data_path)
    print(f"Tokenized dataset saved to {tokenized_data_path}")


class PaperDataset(Dataset):
    def __init__(
        self,
        csv_path,
        label_list,
        tokenizer_name="bert-base-uncased",
        max_length=512,
        tokenized_data_path="tokenized_data.pt",
        cache_dir=None,
        create_new_dataset=False,
        limit=None,
    ):
        """
        Args:
            csv_path (str): Path to the CSV file.
            tokenizer_name (str): Name or path of the pretrained tokenizer.
            label_list (list): List of all possible labels (tags).
            max_length (int): Maximum sequence length for tokenization.
            cache_path (str): Path to save/load the tokenized dataset.
            cache_dir (str, optional): Directory to store cached tokenizer files.
        """
        self.tokenized_data_path = tokenized_data_path
        self.label_list = label_list
        self.label_list_length = len(label_list)
        self.label_to_idx = {label: idx for idx, label in enumerate(label_list)}
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.limit = limit
        self._load_tokenized_data()

    def _tokenize_and_save(self, csv_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, cache_dir=self.cache_dir
        )
        # Load CSV data
        df = pd.read_csv(csv_path)
        num_samples = len(df)

        # Preallocate tensors for tokenized data
        self.input_ids = torch.zeros((num_samples, self.max_length), dtype=torch.long)
        self.attention_mask = torch.zeros(
            (num_samples, self.max_length), dtype=torch.long
        )
        self.labels = torch.zeros(
            (num_samples, len(self.label_list)), dtype=torch.float
        )

        # Tokenize documents and encode labels

        # TODO look into max_length. BERT usees 512 !!! is this an issue?
        progress_bar = tqdm(
            total=len(df) if not self.limit else self.limit,
            desc="Extracting tensors from papers & assigning tags",
            unit="papers",
        )
        for index, row in df.iterrows():
            if index > self.limit:
                break
            # Tokenize the document
            tokens = self.tokenizer(
                row["doc"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            # these next two lines might cause issues due to squeezing
            # and dimensionalites
            self.input_ids[index] = tokens["input_ids"].squeeze()
            self.attention_mask[index] = tokens["attention_mask"].squeeze()

            # Process tags and store labels
            tags = row["categories"].split(" ")  # Assuming tags are space separated
            for tag in tags:
                if tag in self.label_to_idx:
                    self.labels[index][self.label_to_idx[tag]] = 1.0

            progress_bar.update(1)
        progress_bar.close()

        # Save the tokenized dataset to disk
        tokenized_data = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }
        torch.save(tokenized_data, self.tokenized_data_path)
        print(f"Tokenized dataset saved to {self.tokenized_data_path}")

    def _load_tokenized_data(self):
        # Load the tokenized dataset from disk
        tokenized_data = torch.load(self.tokenized_data_path)
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.labels = tokenized_data["labels"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
