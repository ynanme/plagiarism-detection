import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from typing import List, Dict

from model.tokenizers.tokenizer import Tokenizer
from model.tokenizers.lemmatizer import lemmatize
from model.featurizers.count_vector import CountVectorFeaturizer
from model.featurizers.tfidf import TfidfFeaturizer
from model.classifiers.naive_bayes import NaiveBayesClassifier
from model.classifiers.logistic_regression import LogisticRegressionClassifier
from model.classifiers.perceptron import PerceptronClassifier
from model.pipeline.plagiarism_detector import PlagiarismDetector
from util.util import read_files
from experimentations.experiment_manager import ExperimentManager


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


def experiment_on_learning (scientist, repeats=[1]):
    for repeat in repeats:
        scientist.run_on_all(repeat=repeat)
        scientist.run_on_trained(repeat=repeat)
        scientist.run_on_untrained(repeat=repeat)

def experiment_on_classes (scientist, repeats=[1]):
    for repeat in repeats:
        scientist.run_on_all(filter_on_testing="plagiat", repeat=repeat)
        scientist.run_on_all(filter_on_testing="non_plagiat", repeat=repeat)
        scientist.run_on_trained(filter_on_testing="plagiat", repeat=repeat)
        scientist.run_on_trained(filter_on_testing="non_plagiat", repeat=repeat)
        scientist.run_on_untrained(filter_on_testing="plagiat", repeat=repeat)
        scientist.run_on_untrained(filter_on_testing="non_plagiat", repeat=repeat)

def experiment_on_preprocessing (scientist, repeats=[1]):
    experiment_on_learning(scientist, repeats)


DATA_DIRECTORY = "data/1"
RESULTS_DIRECTORY = "experiments_results/perm"

DETECTORS = [
    PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer()),
    PlagiarismDetector(MultinomialNB(), CountVectorizer())
]
NAMES = ["NB_CV", "NB_CV_Sk"]

DETECTORS_PREPROCESSING = [
    PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer()),
    PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(Tokenizer(stop_words=[]))),
    PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(Tokenizer(lemmatizer=lemmatize)))
]
NAMES_PREPROCESSING = ["NB_CV_SW", "NB_CV_NO_SW", "NB_CV_Lemma"]


DETECTORS_ = [
    PlagiarismDetector(PerceptronClassifier(), CountVectorFeaturizer()),
    PlagiarismDetector(LogisticRegressionClassifier(), CountVectorFeaturizer())
]
NAMES_ = ["Percept_CV", "Logis_CV"]


DETECTORS__ = [
    PlagiarismDetector(NaiveBayesClassifier(), TfidfFeaturizer()),
    PlagiarismDetector(MultinomialNB(), TfidfVectorizer())
]
NAMES__ = ["NB_TFIDF", "NB_TFIDF_Sk"]



if __name__ == "__main__":

    data = load_data(DATA_DIRECTORY)
    data[0], data[1] = data[0][:30], data[1][:30]

    student = ExperimentManager(DETECTORS, NAMES, data, RESULTS_DIRECTORY)
    student_ = ExperimentManager(DETECTORS_, NAMES_, data, RESULTS_DIRECTORY)
    student__ = ExperimentManager(DETECTORS__, NAMES__, data, RESULTS_DIRECTORY)
    inspector = ExperimentManager(DETECTORS_PREPROCESSING, NAMES_PREPROCESSING, data, RESULTS_DIRECTORY)

    #experiment_on_learning(student)
    #experiment_on_classes(student)
    #experiment_on_preprocessing(inspector)

    experiment_on_learning(student_)
    experiment_on_learning(student__)
    #experiment_on_classes(student__)

    #experiment_on_learning(student, [5])
    #experiment_on_classes(student, [5])
    experiment_on_preprocessing(inspector, [5])
