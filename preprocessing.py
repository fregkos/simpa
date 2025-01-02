import os
import nltk
import re
from typing import List
from config import VERBOSE
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import defaults

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Calculate stop words set once
remove_stopwords = set(stopwords.words("english"))

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
    words = [word for word in words if word not in remove_stopwords]

    return words


def create_necessary_folders(parent="."):
    models_path = os.path.join(parent, defaults.MODELS_PATH)
    datasets_path = os.path.join(parent, defaults.DATASETS_PATH)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(datasets_path, exist_ok=True)


def clean_text(text: str) -> str:
    """
    This regex replaces:
    1. Newlines, carriage returns, and tabs [\n\r\t]
    2. LaTeX math expressions between dollar signs \$.*?\$
    3. LaTeX commands like \command{arg} \\[a-zA-Z]+(?:\{.*?\})*
    With a single space
    """
    # pattern = r"[\n\r\t]|\\\(.*?\\\)|\\\[.*?\\\]|\$.*?\$|\\[a-zA-Z]+(?:\{.*?\})*"
    pattern = r'\s+'
    cleaned_text = re.sub(pattern, " ", text).strip()

    if VERBOSE:
        print(text, "\n was cleaned and now is: \n", cleaned_text)

    return cleaned_text
