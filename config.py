import argparse
from pprint import pprint
import defaults

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        help="the input file path of the dataset",
        # required=True,
        default=defaults.DATASET_PATH,
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="the output file path of the dataset, with the embeddings saved",
        default=defaults.CSV_PATH,
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="number of sequential lines to parse from the input file",
    )
    parser.add_argument(
        "-f",
        "--fields",
        type=str,
        nargs="+",
        default=["title", "abstract", "categories"],
        help="fields to extract from the dataset (order matters)",
    )
    parser.add_argument(
        "-m",
        "--model_file",
        type=str,
        help="the model file path trained on the given dataset",
        default=defaults.DOC2VEC_MODEL_PATH,
    )
    parser.add_argument(
        "-c",
        "--clean_text",
        action=argparse.BooleanOptionalAction,
        help="clean abstract while preprocessing ('abstract' must be in the FIELDS)",
        default=True,
    )
    parser.add_argument(
        "-nd",
        "--new_dataset",
        action=argparse.BooleanOptionalAction,
        help="(re)build the dataset from scratch, even if it exists",
        default=False,
    )
    parser.add_argument(
        "-nm",
        "--new_model",
        action=argparse.BooleanOptionalAction,
        help="(re)create the Doc2Vec model from scratch, even if it exists",
        default=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        help="print verbose output of the actions that are occuring",
        default=False,
    )
    parser.add_argument(
        "-t",
        "--transformer",
        action=argparse.BooleanOptionalAction,
        help="use transformer instead of doc2vec for classifying the documents",
        default=False,
    )
    parser.add_argument(
        "-ee",
        "--extract_embeddings",
        action=argparse.BooleanOptionalAction,
        help="extract embeddings from dataset anew (even if they already exist) for faster classification",
        default=False,
    )
    return parser.parse_args()


def print_active_arguments(args: argparse.Namespace):
    print("Active arguments:")
    pprint(vars(args))


args = parse_arguments()
VERBOSE = args.verbose

if VERBOSE:
    print_active_arguments(args)
