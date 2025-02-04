import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
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


def compare3 (classs):
    detectors = {
        "inspector": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer()),
        "director": PlagiarismDetector(MultinomialNB(), CountVectorizer())
    }
    items = list(CATEGORIES.items())
    targets = [i for i in range(len(items)) if items[i][1] == classs]
    CATEGORIESS = dict([items[i] for i in targets])
    DATAA = [DATA[i] for i in targets]
    accuracies = []
    for limit in range(2,len(DATAA)+1):
        data = DATAA[:limit]
        categories = dict(list(CATEGORIES.items())[:limit])
        train_detectors(list(detectors.values()), data, categories)
        accuracies.append(evaluate_detectors(list(detectors.values()), data, categories))
    plt.figure()
    plt.plot([_ for _ in range(1,len(DATAA))], [accuracy[0] for accuracy in accuracies])
    plt.plot([_ for _ in range(1,len(DATAA))], [accuracy[1] for accuracy in accuracies])
    plt.savefig(f"fig3{classs}.png")
    plt.show()


def partition (elements, weights, partition_on_indexes=False):
    if math.isclose(sum(weights), 1):
        all_count = len(elements)
        elements_ = elements.copy() if not partition_on_indexes else [index for index in range(all_count)]
        parts = []
        for weight in weights:
            count = round(weight * all_count)
            part = []
            for _ in range(count):
                if len(elements_) > 0:
                    element = random.choice(elements_)
                    part.append(element)
                    elements_.remove(element)
            if len(part) > 0:
                parts.append(part)
        return parts
    return partition(elements, weights + [1-sum(weights)])


def draw_plot (
    X, Ys,
    plot_labels = [], axis_labels = ["X", "Y"], title = "Plots",
    show_figure = True, saving_figure_path = None
):
    plt.figure()
    if plot_labels == []:
        plot_labels = ["Plot " + str(i) for i in range(1,len(Xs)+1)]
    for Y, label in zip(Ys, plot_labels):
        plt.plot(X, Y, label=label)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.legend()
    if saving_figure_path is not None:
        plt.savefig(saving_figure_path)
    if show_figure:
        plt.show()



if __name__ == "__main__":

    compare()
    compare2()
    compare3("plagiat")
    compare3("non_plagiat")

    """
    elements = [i for i in range(15)]
    elements = [47,20,1,-5,45]
    print(partition(elements, [0.3,0.4]))
    print(elements)
    

    n, m, a, b = 3, 10, -4, 15
    draw_plot(
        [[random.randint(a,b) for _ in range(m)] for _ in range(n)],
        [[random.randint(a,b) for _ in range(m)] for _ in range(n)],
        ["le renard", "le loup", "le lièvre"], ["la vie", "la mort"], "Les animaux",
        True, "test"
    )
    """