"""
This file contains default values we need in the project.
"""

import os

DATASETS_PATH = "datasets"
MODELS_PATH = "models"

DATASET_PATH = os.path.join(DATASETS_PATH, "arxiv-metadata-oai-snapshot.json")
DOC2VEC_MODEL_PATH = os.path.join(MODELS_PATH, "doc2vec.model")
LINESENTENCES_PATH = os.path.join(DATASETS_PATH, "line_sentences.txt")
CSV_PATH = os.path.join(DATASETS_PATH, "tagged_line_docs.csv")
LABELS_PATH = os.path.join(DATASETS_PATH, "labels.txt")
TOKENIZED_DATA_PATH = os.path.join(DATASETS_PATH, "tokenized_data.pt")
