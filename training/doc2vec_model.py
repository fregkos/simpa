import logging
import os
from multiprocessing import cpu_count
from typing import Iterable

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

from defaults import CSV_PATH, MODELS_PATH
from training.tagged_doc_line_iterator import TaggedDocumentLineIterator

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def train_doc2vec_model(documents: Iterable[TaggedDocument]) -> Doc2Vec:
    """
    Trains a Doc2Vec model on the given tagged data.

    :param tagged_data: A list of TaggedDocument objects.
    :return: The trained Doc2Vec model.
    """
    model = Doc2Vec(
        documents=documents,
        workers=cpu_count(),
        vector_size=256,
        window=10,
        min_count=2,
        epochs=10,
        alpha=0.025,
        min_alpha=0.001,
        negative=10,
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


def train_or_load_classification_model(embedding_model: Doc2Vec):
    model_path = os.path.join(MODELS_PATH, "model.keras")
    if os.path.exists(model_path):
        return load_model("model.keras"), None

    data = pd.read_csv(CSV_PATH, dtype=str)
    columns_to_keep = ["paper_id", "hyperclasses"]
    columns_to_drop = [
        col for col in data.columns if col not in columns_to_keep
    ]  # ["categories", "title", "abstract", "preprocessed_doc"]
    data.drop(columns_to_drop, axis=1, inplace=True)
    data["hyperclasses"] = data["hyperclasses"].apply(lambda x: x.split(" "))
    data["vector"] = data["paper_id"].apply(lambda x: embedding_model.dv.get_vector(x))

    data.head()

    # TODO: Get those dynamically
    labels = set(
        [
            "cs",
            "econ",
            "eess",
            "math",
            "physics",
            "q-bio",
            "q-fin",
            "stat",
        ]
    )
    # Initialize MultiLabelBinarizer for multi-label encoding
    mlb = MultiLabelBinarizer().fit([labels])
    print(mlb.classes_)

    vectors = np.vstack(data["vector"].values)
    X_train, X_test, y_train, y_test = train_test_split(
        vectors, data["hyperclasses"], test_size=0.2, random_state=42, shuffle=True
    )

    y_train_bin = mlb.transform(y_train)
    y_test_bin = mlb.transform(y_test)

    # Define a simple neural network model
    model = Sequential(
        [
            Dense(128, input_dim=X_train.shape[1], activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(y_train_bin.shape[1], activation="sigmoid"),
        ]
    )

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(
        X_train,
        y_train_bin,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test_bin),
    )

    model.save("model.keras")

    # Evaluate the model on the test set
    y_pred_bin = (model.predict(X_test) > 0.5).astype(int)

    print("Hamming Loss:", np.mean(np.sum(y_pred_bin != y_test_bin, axis=1)))
    print(
        "Classification Report:\n",
        classification_report(y_test_bin, y_pred_bin, target_names=mlb.classes_),
    )

    return model, history
