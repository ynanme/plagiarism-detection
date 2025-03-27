import os
import pandas as pd
from typing import List, Dict

from util.util import read_files


def map_value(value: str) -> str:
    return 1 if value in ['heavy', 'cut', 'light'] else 0

def load_labels(folder_path: str) -> Dict[str, str]:
    expected_results = pd.read_csv(os.path.join(folder_path, "file_information.csv"), sep=",")
    labels = dict(zip(expected_results["File"], expected_results['Category']))
    return {k: map_value(v) for k, v in sorted(labels.items())}

def load_data (directory):
    labels = load_labels(directory)
    texts, files_names = read_files(directory, files_names_required=True)
    categories = [labels[file_name] for file_name in files_names]
    for file_name, category in zip(files_names, categories):
        assert labels[file_name] == category
    return [texts, categories]

def ada_boost_labels_mapping_function (label, reverse=False):
    if reverse:
        if label == -1:
            return 1
        return 0
    if label == 1:
        return -1
    return 1