import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Dict, List, Any

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

def preprocess_data(dataset: Dict[str, Any], fields: List[str]) -> List[str]:
    """
    Preprocesses the dataset by concatenating specified fields into a single document,
    converting to lowercase, tokenizing, and removing stop words.

    Args:
        dataset (Dict[str, Any]): A dictionary where keys are paper IDs and values are dictionaries
                                  containing various fields of data.
        fields (List[str]): A list of field names to concatenate for each paper.

    Returns:
        List[str]: A list of preprocessed words from the concatenated document.
    """
    
    for paper_id in dataset.keys():
        # Create a concatenated doc based on all the given fields, delimited by space
        doc = " ".join([dataset[paper_id][field] for field in fields])

        # Convert the entire document to lower case
        doc = doc.lower()

        # Tokenize the words in the document
        words = word_tokenize(doc)

        # Remove stop words from the document
        remove_stopwords = set(stopwords.words("english"))
        words = [word for word in words if word not in remove_stopwords]

        return words
