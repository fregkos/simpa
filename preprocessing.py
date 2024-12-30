from typing import Any, Dict, List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")


def preprocess_data(document: str) -> List[str]:
    """
    Preprocesses the dataset by concatenating specified fields into a single document,
    converting to lowercase, tokenizing, and removing stop words.

    Args:
        document (str): The original document to be preprocessed.

    Returns:
        List[str]: A list of preprocessed words from the concatenated document.
    """

    # Convert the entire document to lower case
    document = document.lower()

    # Tokenize the words in the document
    words = word_tokenize(document)

    # Remove stop words from the document
    remove_stopwords = set(stopwords.words("english"))
    words = [word for word in words if word not in remove_stopwords]

    return words
