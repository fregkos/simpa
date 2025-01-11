import csv
from gensim.models.doc2vec import TaggedDocument


class TaggedDocumentLineIterator:
    """
    This class reads a CSV file containing tagged line documents and yields TaggedDocuments.
    Each document is represented as a list of words with associated tags.
    Args:
        filepath (str): The path to the CSV file.
        delimiter (str): The delimiter used in the CSV file. Default is " ".
    """

    def __init__(self, filepath, delimiter=" "):
        self.filepath = filepath
        self.delimiter = delimiter

    def __iter__(self):
        with open(self.filepath, "r", newline="") as file:
            reader = csv.reader(file)

            # Skip the header row
            next(reader)

            for row in reader:
                paper_id, categories, doc = row
                categories = categories.split(" ")
                tags = [paper_id] + categories
                text = doc.split(" ")
                yield TaggedDocument(text, tags=tags)
