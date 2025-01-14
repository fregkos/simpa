"""
Authors: Bouzianas Nikoloaos, Fregkos Periklis, Gogos Lazaros
Year: 2024-2025
Task: NLP Project - Scientific Paper Search Engine based on keywords
Data & Web Science - Aristotle University of Thessaloniki
"""

import os

import defaults
from config import parse_arguments
from data_loader import extract_fields, preprocess_and_save_dataset_as_csv, read_labels
from defaults import CSV_PATH, categorized_labels
from training import hierarchical_paper_dataset
from training.doc2vec_model import train_or_load_classification_model
from training.hierarchical_transformer_model import HierarchicalClassifier
from training.preprocessing import create_necessary_folders


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    limit = args.limit
    input_file_path = args.input_file
    model_file_path = args.model_file
    new_dataset = args.new_dataset
    create_new_model = args.new_model
    use_transformer = args.transformer
    extract_embeddings = args.extract_embeddings

    # Firstly, create necessary directories if needed
    create_necessary_folders()

    if not os.path.exists(CSV_PATH) or new_dataset:
        fields = ["title", "abstract", "categories"]
        dataset = extract_fields(input_file_path, fields, limit)
        preprocess_and_save_dataset_as_csv(dataset, CSV_PATH)

    # 1. Import Doc2Vec and create a new model if it doesn't exist
    if not use_transformer:  # this should probably be placed elsewhere for clarity
        history, model = train_or_load_classification_model(
            model_file_path, create_new_model
        )
    else:  # if use_transformer
        label_list = read_labels(defaults.LABELS_PATH)

        # checks are performed inside this following class
        # Check if tokenized data exists on disk
        if (
            os.path.exists(defaults.TOKENIZED_DATA_PATH)
            and os.path.exists(CSV_PATH)
            and not new_dataset
        ):
            print(
                f"Dataset already tokenized, at {defaults.HIERARCHICAL_TOKENIZED_DATA_PATH}..."
            )
        else:
            print("Tokenizing dataset...")
            hierarchical_paper_dataset.prepare_and_tokenize_dataset(
                CSV_PATH,
                label_list,
                hyperclasses=list(categorized_labels.keys()),
                categorized_labels=categorized_labels,
            )

        classifier = HierarchicalClassifier(
            CSV_PATH,
            hyperclass_list=list(categorized_labels.keys()),
            label_list=[
                label for labels in categorized_labels.values() for label in labels
            ],
            hyperclass_to_label_map=categorized_labels,
            batch_size=352,  # it's just tensors after all
            lr=8e-2,
            threshold=0.5,
        )

        # if the embeddings have already been extracted
        # or if we were instructed to extract them
        if (
            not os.path.exists(defaults.HIERARCHICAL_EMBEDDINGS_DATA_PATH)
            or extract_embeddings
        ):
            classifier.extract_and_save_embeddings()  # this should be done only once!

        classifier.train()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
