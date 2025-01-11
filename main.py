"""
Authors: Bouzianas Nikoloaos, Fregkos Periklis, Gogos Lazaros
Year: 2024-2025
Task: NLP Project - Scientific Paper Search Engine based on keywords
Data & Web Science - Aristotle University of Thessaloniki
"""

import os
from pprint import pprint

from keybert import KeyBERT

from config import parse_arguments
from data_loader import (
    extract_fields,
    load_dataset,
    preprocess_and_save_dataset_as_csv,
    read_labels,
)
from defaults import CSV_PATH
from training.doc2vec_model import train_or_load_model
from training import transformer_model
from training.preprocessing import create_necessary_folders, preprocess_data
from gensim.models.doc2vec import Doc2Vec
import defaults

from training.paper_dataset import PaperDataset


def main(args):
    limit = args.limit
    fields = args.fields

    input_file_path = args.input_file
    dataset_file_path = args.output_file
    model_file_path = args.model_file
    clean_text = args.clean_text

    new_dataset = args.new_dataset
    create_new_model = args.new_model
    use_transformer = args.transformer
    dataset = {}

    # Firstly, create necessary directories if needed
    create_necessary_folders()

    # Check if the pruned file exists before loading it
    # And make sure the user has not asked for rebuilding the dataset from scratch
    if (
        os.path.exists(dataset_file_path)
        and os.path.exists(CSV_PATH)
        and not new_dataset
    ):
        dataset = load_dataset(dataset_file_path)
    else:
        # otherwise, load the full dataset and prune it
        dataset = extract_fields(input_file_path, fields, limit, clean_text)
        # save_dataset(dataset) TODO ?? shouldnt this be here? otherwise where is the dataset
        # for it to be loaded the next time ? we now save the CSV ! @Periklis
        preprocess_and_save_dataset_as_csv(dataset, CSV_PATH)

    # 1. Import Doc2Vec and create a new model if it doesn't exist
    if not use_transformer:  # this should probably be placed elsewhere for clarity
        model = train_or_load_model(CSV_PATH, model_file_path, create_new_model)

        # 2. create KeyBERT model for keyword extraction
        keybert_model = KeyBERT()

        # 2. Find top N similar papers by asking a query from the user
        top_n_results = 5
        find_similar_papers(dataset, model, top_n_results, keybert_model)
    else:  # if use_transformer
        label_dict = read_labels(defaults.LABELS_PATH)

        label_list = label_dict.keys()
        # checks are performed inside this following class
        dataset = PaperDataset(
            CSV_PATH,
            label_list,
            tokenized_data_path=defaults.TOKENIZED_DATA_PATH,
            create_new_dataset=new_dataset,
            limit=limit,
        )
        model = transformer_model.TransformerClassifier(
            dataset,
            create_new_model=create_new_model,
            epochs = 50,
            lr = 1e-3,
            batch_size=128,
            n_neurons=256,
            labels=dataset.label_list_length
        )

        # model = transformer_model.train_transformer_model(CSV_PATH)


def find_similar_papers(
    dataset: dict, model: Doc2Vec, top_n_results: int, keybert_model: KeyBERT
):

    while True:
        query = input("Enter a query: ")
        if query in ["/exit", "/e", "/quit", "/q"]:
            break
        preprocessed_query = preprocess_data(query)
        query_vector = model.infer_vector(preprocessed_query)

        similar_docs = model.dv.most_similar([query_vector], topn=top_n_results)
        print(f"{similar_docs=}")

        # Example
        # Compare a paper with it's categories
        print("\n\n\nExample: Compare a paper with it's categories")
        paper_id = "0704.0001"
        paper_vs_self_categoies = model.dv.similarity(
            paper_id, dataset[paper_id]["categories"]
        )
        print(
            f"Paper ID: {paper_id},\ncategories: {dataset[paper_id]['categories']},\nSimilarity to self categories: {paper_vs_self_categoies}"
        )
        print("\n" * 3)

        for paper_id, similarity in similar_docs:
            try:
                print(f"Paper ID: {paper_id}, Similarity: {similarity}")
                pprint(dataset[paper_id]["title"])
                pprint(dataset[paper_id]["abstract"])
                print("\n")

            except KeyError:
                print(
                    f"The category {paper_id} is {similarity} similar to your query.\n"
                )
                continue

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
