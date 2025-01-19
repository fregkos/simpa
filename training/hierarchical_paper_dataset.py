import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os
import defaults
import torch.nn.functional as F


# Data cleanup for tags that are not available in the preZ categories
def _fix_tag(tag):
    if tag == "chem-ph":
        tag = "physics.chem-ph"
    elif tag == "plasm-ph":
        tag = "physics.plasm-ph"
    elif tag == "mtrl-th":
        tag = "cond-mat.mtrl-th"
    elif tag == "atom-ph":
        tag = "physics.atom-ph"
    elif tag == "comp-gas":
        tag = "nlin.CG"
    elif tag == "cmp-lg":
        tag = "cs.CL"
    elif tag == "funct-an":
        tag = "math.FA"
    elif tag == "adap-org":
        tag = "nlin.AO"
    elif tag == "acc-phys":
        tag = "nlin.CD"
    elif tag == "ao-sci":
        tag = "physics.ao-ph"
    elif tag == "patt-sol":
        tag = "nlin.PS"
    elif tag == "solv-int":
        tag = "nlin.SI"
    elif tag == "supr-con":
        tag = "cond-mat.supr-con"
    elif tag == "bayes-an":
        tag = "physics.data-an"
    elif tag == "q-alg":
        tag = "math.QA"
    elif tag == "dg-ga":
        tag = "math.DG"

    return tag


def prepare_and_tokenize_dataset(
    csv_path,
    label_list,
    hyperclasses,
    categorized_labels,
    tokenizer_name="bert-base-uncased",
    cache_dir=None,
    max_length=512,
    tokenized_data_path=defaults.HIERARCHICAL_TOKENIZED_DATA_PATH,
    limit=None,
):
    """Prepare and tokenize the dataset, saving both hyperclass and detailed labels"""
    df = pd.read_csv(csv_path)
    num_samples = len(df)
    label_to_idx = {label: idx for idx, label in enumerate(label_list)}
    # Initialize tensors for both hyperclass and detailed labels
    hyperclass_labels = torch.zeros((num_samples, len(hyperclasses)), dtype=torch.float)
    detailed_labels = torch.zeros((num_samples, len(label_to_idx)), dtype=torch.float)
    label_to_hyperclass = {}
    for hyperclass, labels in categorized_labels.items():
        for label in labels:
            label_to_hyperclass[label] = hyperclass
    hyperclass_to_idx = {hclass: idx for idx, hclass in enumerate(hyperclasses)}
    progress_bar = tqdm(
        total=len(df),
        desc="Processing papers with hierarchical labels",
        unit="papers",
    )

    unrecognized_labels = set()
    for index, row in df.iterrows():
        # Process tags
        tags = row["categories"].split(" ")

        # Track which hyperclasses are active for this paper
        active_hyperclasses = set()
        for tag in tags:
            if tag in label_to_hyperclass:
                tag = _fix_tag(tag)
                # Set detailed label
                detailed_labels[index][label_to_idx[tag]] = 1.0

                # Add hyperclass
                hyperclass = label_to_hyperclass[tag]
                active_hyperclasses.add(hyperclass)
            else:
                unrecognized_labels.add(tag)  # these cause no problem
                # and are addressed in _fix_tag()
        # Set hyperclass labels
        for hyperclass in active_hyperclasses:
            hyperclass_labels[index][hyperclass_to_idx[hyperclass]] = 1.0

        progress_bar.update(1)

    progress_bar.close()
    if len(unrecognized_labels) != 0:
        print(f"{unrecognized_labels=}")
    # Save the processed dataset
    tokenized_data = {
        "hyperclass_labels": hyperclass_labels,
        "detailed_labels": detailed_labels,
    }

    torch.save(tokenized_data, tokenized_data_path)
    print(f"Hierarchical dataset saved to {tokenized_data_path}")



class HierarchicalPaperDataset(Dataset):
    def __init__(
        self,
        csv_path,
        categorized_labels,
        max_length=512,
        tokenized_data_path=defaults.HIERARCHICAL_TOKENIZED_DATA_PATH,
        cache_dir=None,
    ):
        """
        Args:
            csv_path: Path to the CSV file containing the papers
            categorized_labels: Dictionary mapping hyperclasses to their labels
            max_length: Maximum sequence length for tokenization
            tokenized_data_path: Path to save/load the tokenized dataset
            cache_dir: Directory to store cached tokenizer files
        """
        self.tokenized_data_path = tokenized_data_path
        self.categorized_labels = categorized_labels
        self.hyperclasses = list(categorized_labels.keys())

        # Create mappings for both hyperclasses and detailed labels
        self.hyperclass_to_idx = {
            hclass: idx for idx, hclass in enumerate(self.hyperclasses)
        }

        # Flatten all labels and create mapping
        all_labels = []
        for labels in categorized_labels.values():
            all_labels.extend(labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

        self.cache_dir = cache_dir
        self.max_length = max_length
        self.csv_path = csv_path

        # Create reverse mapping from label to hyperclass
        self.label_to_hyperclass = {}
        for hyperclass, labels in categorized_labels.items():
            for label in labels:
                self.label_to_hyperclass[label] = hyperclass

        # if not os.path.exists(tokenized_data_path):
        #     prepare_and_tokenize_dataset()
        self._load_tokenized_data() #should be called after
        self.embeddings = 0
    def _load_tokenized_data(self):
        """Load the processed dataset from disk"""
        tokenized_data = torch.load(self.tokenized_data_path)
        self.hyperclass_labels = tokenized_data["hyperclass_labels"]
        self.detailed_labels = tokenized_data["detailed_labels"]
        df = pd.read_csv(self.csv_path)
        self.documents = df["doc"]
        # this should be done from this file ? but we cant, as we have to load the trunk and the tokenizer ?

    def _load_embeddings(self, embedding_path=defaults.HIERARCHICAL_EMBEDDINGS_DATA_PATH):
        self.embeddings = torch.load(embedding_path)


    # hamming loss
    # 0 1 1 0 target
    # 0 0 0 0 predicted
    # 0.5 loss

    # 0 1 0 0 target
    # 0 0 1 1 predicted
    # 0.75 loss

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        """Get both hyperclass and detailed labels for a document"""
        # Get the document and its labels
        doc = "classification: "+ self.documents[idx]
        hyperclass_label = self.hyperclass_labels[idx]
        detailed_label = self.detailed_labels[idx]

        # Get the active hyperclass name(s)
        active_hyperclasses = [
            self.hyperclasses[i]
            for i in range(len(self.hyperclasses))
            if hyperclass_label[i] == 1
        ]
        embedding = self.embeddings[idx] if isinstance(self.embeddings, torch.Tensor) else torch.zeros((1,1))
        return {
            "doc": doc,
            "hyperclass_labels": hyperclass_label,
            "detailed_labels": detailed_label,
            "hyperclass": (
                active_hyperclasses[0] if active_hyperclasses else None
            ),  # Primary hyperclass
            "embedding": embedding,
        }

    def get_label_counts(self):
        """Get counts of hyperclasses and detailed labels in the dataset"""
        hyperclass_counts = torch.sum(self.hyperclass_labels, dim=0)
        detailed_counts = torch.sum(self.detailed_labels, dim=0)

        hyperclass_stats = {
            self.hyperclasses[i]: int(hyperclass_counts[i])
            for i in range(len(self.hyperclasses))
        }

        detailed_stats = {
            label: int(detailed_counts[self.label_to_idx[label]])
            for label in self.label_to_idx
        }

        return hyperclass_stats, detailed_stats

    