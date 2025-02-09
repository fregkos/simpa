import gc
import logging
import os
import plot
from multiprocessing import cpu_count
from pprint import pprint
from typing import Iterable

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from focal_loss import BinaryFocalLoss

from defaults import CSV_PATH, MODELS_PATH
from training.preprocessing import fix_tag_hyperclass
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


# def train_or_load_classification_model(embedding_model: Doc2Vec):
def train_or_load_classification_model(model_file_path, new_doc2vec_model):
    model_path = os.path.join(MODELS_PATH, "model.keras")
    if os.path.exists(model_path):
        print("Model already exists. Loading existing model...")
        return keras.models.load_model("model.keras"), None
    else:
        print("Model does not exist. Training classification model...")

    # 1. Import Doc2Vec and create a new model if it doesn't exist
    embedding_model = train_or_load_model(CSV_PATH, model_file_path, new_doc2vec_model)

    data = pd.read_csv(CSV_PATH, dtype=str)

    columns_to_keep = ["paper_id", "hyperclasses"]
    columns_to_drop = [
        col for col in data.columns if col not in columns_to_keep
    ]  # ["categories", "title", "abstract", "preprocessed_doc"]
    data.drop(columns_to_drop, axis=1, inplace=True)

    # Select randomly X data
    WANTED_VALUES = 500_000
    np.random.seed(42)
    random_indices = np.random.choice(len(data), WANTED_VALUES, replace=False)
    data = data.iloc[random_indices]

    data["hyperclasses"] = data["hyperclasses"].apply(lambda x: x.split(" "))
    data["hyperclasses"] = data["hyperclasses"].apply(
        lambda x: [fix_tag_hyperclass(tag) for tag in x]
    )  # TODO: Remove it and preprocess the csv again.

    data["vector"] = data["paper_id"].apply(lambda x: embedding_model.dv.get_vector(x))

    data.head()

    # TODO: Get those dynamically
    labels = [
        "cs",
        "econ",
        "eess",
        "math",
        "physics",
        "q-bio",
        "q-fin",
        "stat",
    ]

    X = np.vstack(data["vector"].values)

    del embedding_model
    data.drop("vector", axis=1, inplace=True)
    gc.collect()

    # Initialize MultiLabelBinarizer for multi-label encoding
    mlb = MultiLabelBinarizer().fit([labels])
    print(mlb.classes_)

    y_bin = mlb.transform(data["hyperclasses"])

    # Split the dataset into training and testing sets using stratifiedKFold
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_indices = []
    test_indices = []

    # Perform stratified splitting
    for fold, (train_index, test_index) in enumerate(mskf.split(X, y_bin), 1):
        # Collect one test set (can choose any fold for final evaluation)
        if fold == 1:
            test_indices = test_index
            print(f"Test index: {test_index}")

        # Collect training indices
        train_indices.extend(train_index)
        print(f"Train index: {train_index}")

    train_indices = np.unique(train_indices)
    test_indices = np.unique(test_indices)

    # Create full training data
    X_train_stratified = X[train_indices]
    y_train_stratified = y_bin[train_indices]

    X_test_stratified = X[test_indices]
    y_test_stratified = y_bin[test_indices]

    np.save("X_train_stratified.npy", X_train_stratified)
    np.save("y_train_stratified.npy", y_train_stratified)
    np.save("X_test_stratified.npy", X_test_stratified)
    np.save("y_test_stratified.npy", y_test_stratified)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     vectors, data["hyperclasses"], test_size=0.2, random_state=42, shuffle=True
    # )

    class_weights = get_class_weights(data, instance=X_train_stratified, labels=labels)

    # Define a simple neural network model
    model = keras.models.Sequential(
        [
            keras.layers.Dense(
                128, input_dim=X_train_stratified.shape[1], activation="relu"
            ),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            # keras.layers.Dense(y_train_stratified.shape[1], activation="sigmoid"),
            keras.layers.Dense(y_train_stratified.shape[1]),
            keras.layers.Activation(keras.activations.softmax),
        ]
    )

    METRICS = [
        # keras.metrics.BinaryCrossentropy(name="bce"),  # same as model's loss
        # keras.metrics.MeanSquaredError(name="Brier score"),
        # keras.metrics.TruePositives(name="tp"),
        # keras.metrics.FalsePositives(name="fp"),
        # keras.metrics.TrueNegatives(name="tn"),
        # keras.metrics.FalseNegatives(name="fn"),
        # keras.metrics.BinaryAccuracy(name="accuracy"),
        # keras.metrics.Precision(name="precision"),
        # keras.metrics.Recall(name="recall"),
        # keras.metrics.AUC(name="auc"),
        # keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.F1Score(average="micro"),
        keras.metrics.F1Score(average="macro"),
        keras.metrics.AUC(name="auc", multi_label=True, num_labels=8),
        keras.metrics.AUC(name="prc", curve="PR", multi_label=True, num_labels=8),
        keras.metrics.Accuracy(name="accuracy"),
    ]

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        # loss=focal_loss(alpha=0.25, gamma=2.0),
        # loss=BinaryFocalLoss(gamma=2),
        metrics=METRICS,
    )

    # Train the model
    history = model.fit(
        X_train_stratified,
        y_train_stratified,
        epochs=5,
        batch_size=32,
        validation_data=(X_test_stratified, y_test_stratified),
        class_weight=class_weights,
    )

    model.save("model.keras")

    logits = tf.keras.Sequential(model.layers[:4]).predict(X_test_stratified)
    temprature = 10
    new_logits = logits / temprature
    y_pred_bin = np.array([tf.nn.softmax(l) for l in new_logits])
    pprint(y_pred_bin)

    # Evaluate the model on the test set
    # y_pred_bin = (model.predict(X_test_stratified) > 0.15).astype(int)
    y_pred_bin = (y_pred_bin > 0.15).astype(int)

    print(
        "Classification Report:\n",
        classification_report(y_test_stratified, y_pred_bin, target_names=mlb.classes_),
    )
    print(
        "Confusion Matrix:\n",
        multilabel_confusion_matrix(
            y_test_stratified,
            y_pred_bin,
            labels=mlb.classes_,
        ),
    )

    # 3. Plot the training history
    plot_history(history)
    plot.plot_confusion_matrix(y_test_stratified, y_pred_bin)
    plot.plot_micro_average_precision(y_test_stratified, y_pred_bin)

    return model, history


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Compute the cross-entropy
        ce_loss = -y_true * tf.math.log(y_pred)

        # Compute the focal loss
        focal_term = tf.pow(1 - y_pred, gamma)
        loss = alpha * focal_term * ce_loss

        # Return the mean loss
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    return loss


def plot_history(history):
    """
    Plots training and validation metrics from the history object.

    :param history: The history object returned by model.fit()
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation accuracy values
    ax[0].plot(history.history["accuracy"])
    ax[0].plot(history.history["val_accuracy"])
    ax[0].set_title("Model accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].legend(["Train", "Validation"], loc="upper left")

    # Plot training & validation loss values
    ax[1].plot(history.history["loss"])
    ax[1].plot(history.history["val_loss"])
    ax[1].set_title("Model loss")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend(["Train", "Validation"], loc="upper left")

    plt.savefig("training_history.png")


# DO NOT IMPORT IN OTHER PLACES
def get_class_weights(data: pd.DataFrame, instance, labels: list):
    for label in labels:
        data[label] = np.array(
            [int(label in hyperclasses) for hyperclasses in data["hyperclasses"]]
        )

    sums = {label: data[label].sum() for label in labels}

    max_class_weight = max(sums.values())
    print(f"max class weight: {max_class_weight}")

    class_weights = {label: max_class_weight / total for label, total in sums.items()}
    pprint(f"class_weights: {class_weights}")

    total_sums = sum(sums.values())
    pprint(f"total sums: {total_sums}")

    for label in labels:
        data.drop(label, axis=1, inplace=True)

    gc.collect()

    return class_weights
