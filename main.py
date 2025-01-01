"""
Authors: Bouzianas Nikoloaos, Fregkos Periklis, Gogos Lazaros
Year: 2024-2025
Task: NLP Project - Scientific Paper Search Engine based on keywords
Data & Web Science - Aristotle University of Thessaloniki
"""

import os
from heapq import heapify, heappop, heappush
from pprint import pprint

from sklearn.metrics.pairwise import cosine_similarity

from config import parse_arguments
from data_loader import extract_fields, load_dataset, save_dataset
from preprocessing import create_necessary_folders, preprocess_data
from training import (
    append_vectors_to_dataset,
    create_or_load_model,
    preprocess_and_tag_documents,
)


def main(args):
    limit = args.limit
    fields = args.fields

    input_file_path = args.input_file
    dataset_file_path = args.output_file
    model_file_path = args.model_file
    clean_abstract = args.clean_abstract

    scratch_dataset = args.scratch_dataset
    scratch_doc2vec_model = args.scratch_model
    dataset = {}

    # Firstly, create necessary directories if need e
    create_necessary_folders()

    # Check if the pruned file exists before loading it
    # And make sure the user has not asked for rebuilding the dataset from scratch
    if os.path.exists(dataset_file_path) and not scratch_dataset:
        dataset = load_dataset(dataset_file_path)
    else:
        # otherwise, load the full dataset and prune it
        dataset = extract_fields(input_file_path, fields, limit, clean_abstract)
        save_dataset(dataset_file_path, dataset)

    # 1. Preprocess and tag documents
    tagged_data = preprocess_and_tag_documents(dataset, fields)

    # 2. Import Doc2Vec and create a new model if it doesn't exist
    model = create_or_load_model(model_file_path, tagged_data, scratch_doc2vec_model)

    # 3. Append vectors to dataset
    append_vectors_to_dataset(dataset, fields, model)

    # 4. Save embeddings to file
    save_dataset(dataset_file_path, dataset, scratch_dataset)

    # 5. Find top N similar papers by asking a query from the user
    top_n_results = 5
    find_similar_papers(dataset, model, top_n_results)


def find_similar_papers(dataset, model, top_n_results):
    heap = []
    heapify(heap)

    while True:
        query = input("Enter a query: ")
        if query in ["/exit", "/e", "/quit", "/q"]:
            break
        preprocessed_query = preprocess_data(query)
        query_vector = model.infer_vector(preprocessed_query)

        for paper_id in dataset.keys():
            heappush(
                heap,
                (
                    -1
                    * cosine_similarity([query_vector], [dataset[paper_id]["vector"]]),
                    paper_id,
                ),
            )

        for _ in range(top_n_results):
            similarity, paper_id = heappop(heap)
            print(f"Paper ID: {paper_id}, Similarity: {-1 * similarity}")
            pprint(dataset[paper_id]["title"])
            pprint(dataset[paper_id]["abstract"])
            print("\n")

    # TODO: Get your input data and preprocess it, then get the embeddings and compare the cosine similarity between all papers in the dataset


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
