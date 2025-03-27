
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

from model.preprocessors.tokenizer import Tokenizer
#from model.preprocessors.lemmatizer import lemmatize
from model.featurizers.featurizer import CustomedFeaturizer
from model.featurizers.count_vector import CountVectorFeaturizer
from model.featurizers.tfidf import TfidfFeaturizer
from model.classifiers.naive_bayes import NaiveBayesClassifier
from model.classifiers.logistic_regression import LogisticRegressionClassifier
from model.classifiers.perceptron import PerceptronClassifier
from model.classifiers.decision_tree import DecisionTree
from model.classifiers.ada_boost import AdaBoost
from model.classifiers.bagging import BaggingClassifier
from model.pipeline.plagiarism_detector import PlagiarismDetector
from util.util import read_files
from experimentations.experiment_manager import *
from experimentations.experiments_utils import *



DATA_DIRECTORY = "data/1"
data = load_data(DATA_DIRECTORY)

def compute_detectors_ranks (training_data_weight, repeat):
    detectors = {
        "naives_bayes": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "perceptron": PlagiarismDetector(PerceptronClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "logistic_regression": PlagiarismDetector(LogisticRegressionClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "decision_tree": PlagiarismDetector(DecisionTree(max_depth=1), CountVectorFeaturizer(), Tokenizer()),
        "ada_boost_stump": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(), Tokenizer()),
        "bagg_stump": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorFeaturizer(), Tokenizer()),
    }
    manager = ExperimentManager(detectors, data)
    scores, rankings = manager.rank_detectors(training_data_weight, repeat)
    Metrics.plot_scores_and_rankings(scores, rankings)
    print(scores)
    print(rankings)


def compute_detectors_learning (training_data_weight, repeat, batch_size_increment):
    detectors = {
        "naives_bayes": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "perceptron": PlagiarismDetector(PerceptronClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "logistic_regression": PlagiarismDetector(LogisticRegressionClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "decision_tree": PlagiarismDetector(DecisionTree(max_depth=1), CountVectorFeaturizer(), Tokenizer()),
        "ada_boost_stump": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(), Tokenizer()),
        "bagg_stump": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorFeaturizer(), Tokenizer()),
    }
    manager = ExperimentManager(detectors, data)
    r = manager.evaluate_detectors_on_learning(training_data_weight, repeat, batch_size_increment)
    Metrics.plot_learning_curves(r[0], batch_size_increment)
    print(r)



if __name__ == "__main__":

    training_data_weight = 0.8
    repeat = 10
    batch_size_increment = 10

    compute_detectors_ranks(training_data_weight, repeat)
    compute_detectors_learning(training_data_weight, repeat, batch_size_increment)

