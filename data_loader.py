import orjson
import os
import subprocess
from typing import Iterator
from tqdm import tqdm
from pprint import pprint
from preprocessing import clean_abstract


def parse_json(json_string: str) -> dict:
    """
    Parses a JSON string into a dictionary.

    Args:
        json_string (str): The JSON string to be parsed.

    Returns:
        dict or None: A dictionary representation of the JSON string, or None if parsing fails.
    """
    try:
        return orjson.loads(json_string)
    except orjson.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        return None


def get_file_lines(file_path: str) -> int:
    """
    Counts the number of lines in a file using platform-specific commands.

    Args:
        file_path (str): The path to the file for which to count lines.

    Returns:
        int: Total number of lines in the file, or -1 if an error occurs.
    """
    total_lines = -1

    # TODO: Untested for Windows
    if os.name == "nt":
        try:
            total_lines = (
                subprocess.check_output(["find", "/c", "/l", file_path])
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to count lines in {file_path}: {e}")
    elif os.name == "posix":
        try:
            total_lines = int(
                subprocess.check_output(["wc", "-l", file_path]).decode().strip().split(' ')[0]
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to count lines in {file_path}: {e}")

    return int(total_lines) if isinstance(total_lines, str) else total_lines


def fetch_data_from_json_file(file_path: str, limit: int = None) -> Iterator[dict]:
    """
    Yields data parsed from a JSON file line by line.

    Args:
        file_path (str): The path to the JSON file.
        limit (int, optional): Maximum number of lines to process. Defaults to None.

    Yields:
        dict: Parsed dictionary representation of each line in the file.
    """
    i = 0
    try:
        with open(file_path, "r") as file:
            for line in file:
                if limit and i >= limit:
                    break
                try:
                    yield parse_json(line)
                    i += 1
                except orjson.JSONDecodeError as e:
                    print(f"Failed to decode JSON in iteration {i}: {e}")

        size = os.path.getsize(file_path) * 1e-9
        print(f"Extraction done, extracted {i} papers, total file size ~{size:.2f} GB")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None


def extract_fields(
    file_path: str,
    fields: list = ["title", "abstract"],
    limit: int = None,
    should_clean_abstract: bool = False,
) -> dict:
    """
    Extract fields from a JSON file. The function reads each line of the file,
    parses it as JSON, and extracts specified fields. It supports cleaning the abstract.

    Args:
        file_path (str, optional): Path to the JSON file. Defaults to "data.json".
        fields (list, optional): List of field names to extract. Defaults to ["title", "abstract"].
        limit (int, optional): Maximum number of lines to process. Defaults to None.
        clean_abstract (bool, optional): Whether to clean the abstract before extraction. Defaults to False.

    Returns:
        dict: A dictionary where keys are paper IDs and values are dictionaries containing the extracted fields.
    """
    dataset = {}
    total_lines = get_file_lines(file_path) if not limit else limit
    
    progress_bar = tqdm(total=total_lines, desc="Loading & Cleaning papers", unit="papers")

    for data in fetch_data_from_json_file(file_path, limit):
        if data is None:
            print("Failed to parse this data. Skipping it.")
            pprint(data)
            continue

        if "abstract" in fields and should_clean_abstract:
            data["abstract"] = clean_abstract(data["abstract"])

        dataset[data["id"]] = {key: data[key] for key in fields if key in data}
        progress_bar.update(1)
    progress_bar.close()

    if limit and limit <= 10:
        pprint(dataset)

    print(f"Extraction done, {len(dataset)}, papers")
    return dataset


def load_dataset(dataset_file_path: str) -> dict:
    """
    Loads a dataset from a JSON file.

    Args:
        dataset_file_path (str): Path to the JSON file containing the dataset.

    Returns:
        dict or None: The loaded dataset as a dictionary, or None if loading fails.
    """
    size = os.path.getsize(dataset_file_path) * 1e-9
    print(f"Loading dataset file: {dataset_file_path}, ~{size:.2f} GB")

    try:
        with open(dataset_file_path, "rb") as f:
            dataset = orjson.loads(f.read())
        print(f"Data loading done, {len(dataset)}, papers")
        return dataset
    except orjson.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        return None


def save_dataset(dataset_file_path: str, dataset: dict):
    """
    Saves a dataset to a JSON file.

    Args:
        dataset_file_path (str): Path where the JSON file will be saved.
        dataset (dict): The dataset to be saved as a dictionary.
    """
    print(f"Dumping current dataset to {dataset_file_path}, please wait...")

    with open(dataset_file_path, "wb") as f:
        f.write(orjson.dumps(dataset, option=orjson.OPT_SERIALIZE_NUMPY))

    size = os.path.getsize(dataset_file_path) * 1e-9
    print(f"Done. Saved to {dataset_file_path}, ~{size:.2f} GB")
