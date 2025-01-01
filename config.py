import argparse
from pprint import pprint


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        help="the input file path of the dataset",
        # required=True,
        default="datasets/arxiv-metadata-oai-snapshot.json",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="the output file path of the dataset, with the embeddings saved",
        default="datasets/dataset.json",
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
        type=list,
        nargs="+",
        default=["title", "abstract"],
        help="fields to extract from the dataset",
    )
    parser.add_argument(
        "-m",
        "--model_file",
        type=str,
        help="the model file path trained on the given dataset",
        default="models/doc2vec.model",
    )
    parser.add_argument(
        "-c",
        "--clean_abstract",
        action=argparse.BooleanOptionalAction,
        help="clean abstract while preprocessing ('abstract' must be in the FIELDS)",
        default=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        help="print verbose output of the actions that are occuring",
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
