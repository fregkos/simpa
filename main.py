"""
Authors: Bouzianas Nikoloaos, Fregkos Periklis, Gogos Lazaros
Year: 2024-2025
Task: NLP Project - Scientific Paper Search Engine based on keywords
Data & Web Science - Aristotle University of Thessaloniki
"""

import os
import argparse
from data_loader import extract_fields, load_dataset, save_dataset
from training import train_and_save_embeddings


def main(args):
    limit = args.limit
    fields = args.fields

    input_file_path = args.input_file
    dataset_file_path = args.output_file

    dataset = {}

    # Check if the pruned file exists before loading it
    if os.path.exists(dataset_file_path):
        dataset = load_dataset(dataset_file_path)
    else:
        # otherwise, load the full dataset and prune it
        dataset = extract_fields(input_file_path, fields, limit)
        save_dataset(dataset_file_path, dataset)

    # The main pipeline, which includes loading the dataset, training embeddings, and saving them.
    train_and_save_embeddings(dataset_file_path, dataset, fields)

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

    args = parser.parse_args()
    main(args)
