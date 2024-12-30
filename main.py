"""
Authors: Bouzianas Nikoloaos, Fregkos Periklis, Gogos Lazaros
Year: 2024-2025
Task: NLP Project - Scientific Paper Search Engine based on keywords
Data & Web Science - Aristotle University of Thessaloniki
"""

import os
import argparse
from data_loader import extract_fields, load_dataset, save_dataset
from training import (
    preprocess_and_tag_documents,
    create_or_load_model,
    append_vectors_to_dataset,
    save_dataset,
)


def main(args):
    limit = args.limit
    fields = args.fields

    input_file_path = args.input_file
    dataset_file_path = args.output_file
    model_path = args.model_path

    dataset = {}

    # Check if the pruned file exists before loading it
    if os.path.exists(dataset_file_path):
        dataset = load_dataset(dataset_file_path)
    else:
        # otherwise, load the full dataset and prune it
        dataset = extract_fields(input_file_path, fields, limit)
        save_dataset(dataset_file_path, dataset)

    # 1. Preprocess and tag documents
    tagged_data = preprocess_and_tag_documents(dataset, fields)

    # 2. Import Doc2Vec and create a new model if it doesn't exist
    model = create_or_load_model(model_path, tagged_data)

    # 3. Append vectors to dataset
    append_vectors_to_dataset(dataset, fields, model)
    
    # 4. Save embeddings to file
    save_dataset(dataset_file_path, dataset)

    # TODO: Get your input data and preprocess it, then get the embeddings and compare the cosine similarity between all papers in the dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        help="the input file path of the dataset",
        # required=True,
        default="datasets/arxiv-metadata-oai-snapshot.json",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="the output file path of the dataset, with the embeddings saved",
        # required=True,
        default="datasets/dataset.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        # default=10,
        help="number of sequential lines to parse from the input file",
    )
    parser.add_argument(
        "--fields",
        type=list,
        nargs="+",
        default=["title", "abstract"],
        help="fields to extract from the dataset",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        help="the model file path trained on the given dataset",
        # required=True,
        default="models/doc2vec.model",
    )

    args = parser.parse_args()
    main(args)
