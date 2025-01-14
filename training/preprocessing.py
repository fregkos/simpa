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

    # Clean and convert the entire document to lower case
    document = clean_text(document.lower())

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
    pattern = r"\s+"
    cleaned_text = re.sub(pattern, " ", text).strip()

    if VERBOSE:
        print(f"{text}\nwas cleaned and now is:\n{cleaned_text}")

    return cleaned_text


def fix_tag(tag):
    if tag == "chem-ph":
        tag = "physics.chem-ph"
    elif tag == "plasm-ph":
        tag = "physics.plasm-ph"
    elif tag == "mtrl-th":
        tag = "cond-mat.mtrl-th"
    elif tag == "atom-ph":
        tag = "physics.atom-ph"
    elif tag == "comp-gas":
        tag = "nlin.CG"
    elif tag == "cmp-lg":
        tag = "cs.CL"
    elif tag == "funct-an":
        tag = "math.FA"
    elif tag == "adap-org":
        tag = "nlin.AO"
    elif tag == "acc-phys":
        tag = "nlin.CD"
    elif tag == "ao-sci":
        tag = "physics.ao-ph"
    elif tag == "patt-sol":
        tag = "nlin.PS"
    elif tag == "solv-int":
        tag = "nlin.SI"
    elif tag == "supr-con":
        tag = "cond-mat.supr-con"
    elif tag == "bayes-an":
        tag = "physics.data-an"
    elif tag == "q-alg":
        tag = "math.QA"
    elif tag == "dg-ga":
        tag = "math.DG"
    elif tag == "alg-geom":
        tag = "math.AG"
    elif tag == "chao-dyn":
        tag = "nlin.CD"

    return tag


def fix_tag_hyperclass(tag: str):
    if tag == "chem-ph":
        tag = "physics"
    elif tag == "plasm-ph":
        tag = "physics"
    elif tag == "mtrl-th":
        tag = "physics"
    elif tag == "atom-ph":
        tag = "physics"
    elif tag == "comp-gas":
        tag = "physics"
    elif tag == "cmp-lg":
        tag = "cs"
    elif tag == "funct-an":
        tag = "math"
    elif tag == "adap-org":
        tag = "physics"
    elif tag == "acc-phys":
        tag = "physics"
    elif tag == "ao-sci":
        tag = "physics"
    elif tag == "patt-sol":
        tag = "physics"
    elif tag == "solv-int":
        tag = "physics"
    elif tag == "supr-con":
        tag = "physics"
    elif tag == "bayes-an":
        tag = "physics"
    elif tag == "q-alg":
        tag = "math"
    elif tag == "dg-ga":
        tag = "math"
    elif tag == "alg-geom":
        tag = "math"
    elif tag == "chao-dyn":
        tag = "phycics"

    return tag
