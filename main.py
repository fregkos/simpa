"""
Authors: Bouzianas Nikoloaos, Fregkos Periklis, Gogos Lazaros
Year: 2024-2025
Task: NLP Project - Scientific Paper Search Engine based on keywords
Data & Web Science - Aristotle University of Thessaloniki
"""

import os

from config import parse_arguments
from data_loader import extract_fields, preprocess_and_save_dataset_as_csv
from defaults import CSV_PATH
from training.doc2vec_model import train_or_load_classification_model
from training.preprocessing import create_necessary_folders


def main(args):
    limit = args.limit
    input_file_path = args.input_file
    model_file_path = args.model_file
    new_dataset = args.new_dataset
    new_doc2vec_model = args.new_model

    # Firstly, create necessary directories if needed
    create_necessary_folders()

    if not os.path.exists(CSV_PATH) or new_dataset:
        fields = ["title", "abstract", "categories"]
        dataset = extract_fields(input_file_path, fields, limit)
        preprocess_and_save_dataset_as_csv(dataset, CSV_PATH)

    # 2. Train the classification model using the Doc2Vec embeddings
    model, history = train_or_load_classification_model(
        model_file_path, new_doc2vec_model
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
