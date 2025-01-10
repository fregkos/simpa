"""
Authors: Bouzianas Nikoloaos, Fregkos Periklis, Gogos Lazaros
Year: 2024-2025
Task: NLP Project - Scientific Paper Search Engine based on keywords
Data & Web Science - Aristotle University of Thessaloniki
"""

import os
import defaults
from pprint import pprint

from config import parse_arguments
from data_loader import extract_fields, load_dataset, save_dataset
from training.preprocessing import create_necessary_folders, preprocess_data
from training.doc2vec_model import train_or_load_model
from keybert import KeyBERT


def main(args):
    limit = args.limit
    fields = args.fields

    input_file_path = args.input_file
    dataset_file_path = args.output_file
    model_file_path = args.model_file
    clean_text = args.clean_text

    new_dataset = args.new_dataset
    new_doc2vec_model = args.new_model
    dataset = {}

    # update defaults
    # ...
    # grab filepath from defaults
    linesentences_file_path = defaults.LINESENTENCES_PATH

    # Firstly, create necessary directories if needed
    create_necessary_folders()

    # Check if the pruned file exists before loading it
    # And make sure the user has not asked for rebuilding the dataset from scratch
    if os.path.exists(dataset_file_path) and not new_dataset:
        dataset = load_dataset(dataset_file_path)
    else:
        # otherwise, load the full dataset and prune it
        dataset = extract_fields(input_file_path, fields, limit, clean_text)
        save_dataset(dataset_file_path, dataset, fields, linesentences_file_path)

    # 1. Import Doc2Vec and create a new model if it doesn't exist
    model = train_or_load_model(
        linesentences_file_path, model_file_path, new_doc2vec_model
    )

    # 2. create KeyBERT model for keyword extraction
    keybert_model = KeyBERT()

    # 2. Find top N similar papers by asking a query from the user
    top_n_results = 5
    find_similar_papers(dataset, model, top_n_results, keybert_model)


def find_similar_papers(
    dataset: dict, model, top_n_results: int, keybert_model: KeyBERT
):

    while True:
        query = input("Enter a query: ")
        if query in ["/exit", "/e", "/quit", "/q"]:
            break
        preprocessed_query = preprocess_data(query)
        query_vector = model.infer_vector(preprocessed_query)

        similar_docs = model.dv.most_similar([query_vector], topn=top_n_results)
        keys = list(dataset.keys())

        for line, similarity in similar_docs:
            paper_id = keys[line]  # THIS IS EXTREMELY SLOW
            print(f"Paper ID: {paper_id}, Similarity: {similarity}")
            pprint(dataset[paper_id]["title"])
            pprint(dataset[paper_id]["abstract"])
            print("\n")

        query_keywords = keybert_model.extract_keywords(
            query,
            keyphrase_ngram_range=(1, 3),
        )

        print("Extracted keywords, based on your query:")
        for keyword, similarity_to_query in query_keywords:
            # delimited by spaces instead of new lines for easy copy-pasting
            print(keyword, end=" ")
        print()  # print empty new line


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
