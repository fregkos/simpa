from typing import Dict, List
from preprocessing import preprocess_data
from data_loader import save_dataset
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint

def preprocess_and_tag_documents(dataset: Dict[str, List], fields: List) -> List[TaggedDocument]:
    """
    Preprocesses the dataset and tags each document with its ID.

    :param dataset: A dictionary where keys are paper IDs and values are lists of words.
    :param fields: The fields to preprocess.
    :return: A list of TaggedDocument objects.
    """
    tagged_data = []
    
    for paper_id in dataset.keys():
        words = preprocess_data(dataset, fields)
        tagged_data.append(TaggedDocument(words, tags=[paper_id]))
    
    return tagged_data

def train_doc2vec_model(tagged_data: List[TaggedDocument]) -> Doc2Vec:
    """
    Trains a Doc2Vec model on the given tagged data.

    :param tagged_data: A list of TaggedDocument objects.
    :return: The trained Doc2Vec model.
    """
    # TODO: Tune hyperparameters for Doc2Vec
    model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    return model

def append_vectors_to_dataset(dataset: Dict[str, List], fields: List, model: Doc2Vec) -> None:
    """
    Appends document vectors to the dataset.

    :param dataset: A dictionary where keys are paper IDs and values are lists of words.
    :param fields: The fields to preprocess.
    :param model: The trained Doc2Vec model.
    """
    for paper_id in dataset.keys():
        words = preprocess_data(dataset, fields)
        dataset[paper_id]["vector"] = model.infer_vector(words)

def train_and_save_embeddings(dataset_file_path: str, dataset: Dict[str, List], fields: List) -> None:
    """
    Trains a Doc2Vec model and appends the document vectors to the dataset. Then saves the embeddings to a file.

    :param dataset_file_path: The path to save the dataset with embeddings.
    :param dataset: A dictionary where keys are paper IDs and values are lists of words.
    :param fields: The fields to preprocess.
    """
    # 1. Preprocess and tag documents
    tagged_data = preprocess_and_tag_documents(dataset, fields) 
    # 2. Train Doc2Vec model
    model = train_doc2vec_model(tagged_data)
    # 3. Append vectors to dataset
    append_vectors_to_dataset(dataset, fields, model)
    # 4. Save embeddings to file
    save_dataset(dataset_file_path, dataset)

    # Print the document vectors (optional)
    # pprint(dataset)