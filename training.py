import concurrent.futures
import logging
import os
from itertools import islice

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

from typing import Dict, List

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

from preprocessing import preprocess_data


def split_dataset(dataset, n):
    """
    Splits the dataset into `n` almost equal parts.

    :param dataset: A dictionary where keys are paper IDs and values are lists of words.
    :param n: The number of parts to split the dataset into.
    :return: A list of datasets, each being a part.
    """
    return [
        dict(
            islice(dataset.items(), i * len(dataset) // n, (i + 1) * len(dataset) // n)
        )
        for i in range(n)
    ]


def preprocess_and_tag_documents(
    dataset: Dict[str, List], fields: List
) -> List[TaggedDocument]:
    """
    Preprocesses the dataset and tags each document with its ID.

    :param dataset: A dictionary where keys are paper IDs and values are lists of words.
    :param fields: The fields to preprocess.
    :return: A list of TaggedDocument objects.
    """
    tagged_data = []

    progress_bar = tqdm(total=len(dataset), desc="Preprocessing papers", unit="papers")

    for paper_id in dataset.keys():
        # Create a concatenated doc based on all the given fields, delimited by space
        document = " ".join([dataset[paper_id][field] for field in fields])
        words = preprocess_data(document)
        tagged_data.append(TaggedDocument(words, tags=[paper_id]))
        progress_bar.update(1)
    progress_bar.close()

    return tagged_data


def train_doc2vec_model(tagged_data: List[TaggedDocument]) -> Doc2Vec:
    """
    Trains a Doc2Vec model on the given tagged data.

    :param tagged_data: A list of TaggedDocument objects.
    :return: The trained Doc2Vec model.
    """
    # TODO: Tune hyperparameters for Doc2Vec (workers = n-1 etc)
    model = Doc2Vec(vector_size=20, min_count=2, epochs=50, dm=0)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    return model


def append_vectors_to_dataset(
    dataset: Dict[str, List], fields: List, model: Doc2Vec
) -> None:
    """
    Appends document vectors to the dataset.

    :param dataset: A dictionary where keys are paper IDs and values are lists of words.
    :param fields: The fields to preprocess.
    :param model: The trained Doc2Vec model.
    """
    progress_bar = tqdm(
        total=len(dataset), desc="Appending vectors to dataset", unit="papers"
    )

    for paper_id in dataset.keys():
        # Create a concatenated doc based on all the given fields, delimited by space
        document = " ".join([dataset[paper_id][field] for field in fields])
        words = preprocess_data(document)
        dataset[paper_id]["vector"] = model.infer_vector(words)
        progress_bar.update(1)
    progress_bar.close()


def train_or_load_model(
    model_path: str, dataset: Dict[str, List], fields: List, new_model: bool = False
) -> Doc2Vec | None:
    """
    Loads the trained Doc2Vec model from the specified path.
    If the model does not exist, it trains a new one using the provided tagged data.
    :param model_path: The path to save the trained model.
    :param tagged_data: A list of TaggedDocument objects.
    :return: A trained Doc2Vec model.
    """
    model = None

    if os.path.exists(model_path) and not new_model:
        print("Model already exists. Loading existing model...")
        model = Doc2Vec.load(model_path)
    else:
        # 0. Preprocess and tag documents
        tagged_data = preprocess_and_tag_documents(dataset, fields)

        print("Training Doc2Vec model...")
        model = train_doc2vec_model(tagged_data)
        model.save(model_path)

    return model
