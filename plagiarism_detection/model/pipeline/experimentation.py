import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from typing import List, Dict

from model.tokenizers.tokenizer import Tokenizer
from model.featurizers.count_vector import CountVectorFeaturizer
from model.classifiers.naive_bayes import NaiveBayesClassifier
from model.pipeline.plagiarism_detector import PlagiarismDetector
from util.util import read_files




def map_value(value: str) -> str:
    return 'plagiat' if value in ['heavy', 'cut'] else 'non_plagiat'

def load_labels(folder_path: str) -> Dict[str, str]:
    expected_results = pd.read_csv(os.path.join(folder_path, "file_information.csv"), sep=",")
    labels = dict(zip(expected_results["File"], expected_results['Category']))
    return {k: map_value(v) for k, v in sorted(labels.items())}


def train_detectors (detectors, data, expected):
    detectors[0].train(data, expected)
    detectors[1].trainn(data, expected)

def evaluate_detectors (detectors, data, expected):
    return [
        detectors[0].evaluate(data, expected),
        detectors[1].evaluatee(data, expected)
    ]


DIRECTORY = "data/1"
DATA = read_files(DIRECTORY)
CATEGORIES = load_labels(DIRECTORY)


def compare ():
    detectors = {
        "inspector": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer()),
        "director": PlagiarismDetector(MultinomialNB(), CountVectorizer())
    }
    accuracies = []
    for limit in range(2,len(DATA)+1):
        data = DATA[:limit]
        categories = dict(list(CATEGORIES.items())[:limit])
        train_detectors(list(detectors.values()), data, categories)
        accuracies.append(evaluate_detectors(list(detectors.values()), data, categories))
    plt.figure()
    plt.plot([_ for _ in range(1,len(DATA))], [accuracy[0] for accuracy in accuracies])
    plt.plot([_ for _ in range(1,len(DATA))], [accuracy[1] for accuracy in accuracies])
    plt.savefig("fig.png")
    plt.show()


def compare2 ():
    detectors = {
        "inspector": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer()),
        "director": PlagiarismDetector(MultinomialNB(), CountVectorizer())
    }
    accuracies = []
    categories_list = list(CATEGORIES.items())
    lim = int(0.8 * len(DATA))
    train_data, test_data = DATA[:lim], DATA[lim:]
    train_cata, test_cata = dict(categories_list[:lim]), dict(categories_list[lim:])
    for limit in range(2,len(train_data)+1):
        data = train_data[:limit]
        categories = dict(list(train_cata.items())[:limit])
        train_detectors(list(detectors.values()), data, categories)
        accuracies.append(evaluate_detectors(list(detectors.values()), test_data, test_cata))
    plt.figure()
    plt.plot([_ for _ in range(1,len(train_data))], [accuracy[0] for accuracy in accuracies])
    plt.plot([_ for _ in range(1,len(train_data))], [accuracy[1] for accuracy in accuracies])
    plt.savefig("fig2.png")
    plt.show()




if __name__ == "__main__":

    compare()
    compare2()