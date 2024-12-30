import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from tqdm import tqdm
from typing import Dict, List

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocessing import preprocess_data


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

    for paper_id in tqdm(dataset.keys(), unit="papers", desc="Preprocessing papers"):
        # Create a concatenated doc based on all the given fields, delimited by space
        document = " ".join([dataset[paper_id][field] for field in fields])
        words = preprocess_data(document)
        tagged_data.append(TaggedDocument(words, tags=[paper_id]))

    return tagged_data


def train_doc2vec_model(tagged_data: List[TaggedDocument]) -> Doc2Vec:
    """
    Trains a Doc2Vec model on the given tagged data.

    :param tagged_data: A list of TaggedDocument objects.
    :return: The trained Doc2Vec model.
    """
    # TODO: Tune hyperparameters for Doc2Vec
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
    for paper_id in dataset.keys():
        # Create a concatenated doc based on all the given fields, delimited by space
        document = " ".join([dataset[paper_id][field] for field in fields])
        words = preprocess_data(document)
        dataset[paper_id]["vector"] = model.infer_vector(words)


def create_or_load_model(
    model_path: str, tagged_data: List[TaggedDocument]
) -> Doc2Vec | None:
    """
    Loads the trained Doc2Vec model from the specified path.
    If the model does not exist, it trains a new one using the provided tagged data.
    :param model_path: The path to save the trained model.
    :param tagged_data: A list of TaggedDocument objects.
    :return: A trained Doc2Vec model.
    """
    model = None

    if os.path.exists(model_path):
        print("Model already exists. Loading existing model...")
        model = Doc2Vec.load(model_path)
    else:
        print("Training Doc2Vec model...")
        model = train_doc2vec_model(tagged_data)
        model.save(model_path)

    return model
