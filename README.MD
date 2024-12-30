# Setup

## Download the arxiv dataset

1. Download the arxiv dataset from here:
    - <https://www.kaggle.com/datasets/Cornell-University/arxiv>
2. Save it under the [datasets](datasets) directory.

## Create a virtual environment and install the required packages

```bash
  python3 -m venv myenv
  source myenv/bin/activate
  pip install -r requirements.txt
```

## Run the application

```bash
  python3 main.py
```

## Usage

```bash
python3 main.py [-h] [--input_file INPUT_FILE] [--output_file OUTPUT_FILE] [--limit LIMIT] [--fields FIELDS [FIELDS ...]]
```

### Command Line Arguments

- `--help`: Show this help message and exit.
- `--input_file INPUT_FILE`: The input file path of the dataset.
- `--output_file OUTPUT_FILE`: The output file path of the dataset, with the embeddings saved.
- `--limit LIMIT`: Number of sequential lines to parse from the input file.
- `--fields FIELDS [FIELDS ...]`: Fields to extract from the dataset.

## Tweaking

### Change the model hyperparameters

You can change the hyperparameters of the model by modifying the [train_doc2vec_model](training.py) function in the [training.py](training.py) script. The main parameters are:

- `vector_size`: Size of the word vectors.
- `min_count`: Minimum number of occurrences for a word to be considered.
- `epochs`: Number of epochs to train the model.
- `dm`: Distributed Memory Model (0 for PV-DM, 1 for DBOW).

You should experiment with more, see <https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html> for more details.

### Change the data preprocessing steps

You can change the data preprocessing steps by modifying the [preprocessing.py](preprocessing.py) script. The main steps are:

- Join given fields into a single string.
- Convert all text to lowercase.
- Tokenize the text into words.
- Remove stop words.

You should experiment with more, see <https://www.nltk.org/> for more details.
