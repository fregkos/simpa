import logging
import os
from multiprocessing import cpu_count
from typing import Iterable
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from training.tagged_doc_line_iterator import TaggedDocumentLineIterator

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


# def train_doc2vec_model(tagged_data: List[TaggedDocument]) -> Doc2Vec:
def train_doc2vec_model(documents: Iterable[TaggedDocument]) -> Doc2Vec:
    """
    Trains a Doc2Vec model on the given tagged data.

    :param tagged_data: A list of TaggedDocument objects.
    :return: The trained Doc2Vec model.
    """
    model = Doc2Vec(
        documents=documents,
        workers=cpu_count(),
        vector_size=50,
        min_count=3,
        epochs=30,
        dm=1,
    )

    return model


def train_or_load_model(
    csv_path: str,
    model_path: str,
    new_model: bool = False,
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
        print("Training Doc2Vec model...")
        model = train_doc2vec_model(TaggedDocumentLineIterator(csv_path))
        model.save(model_path)

    return model


def train_or_load_classification_model():
    pass
