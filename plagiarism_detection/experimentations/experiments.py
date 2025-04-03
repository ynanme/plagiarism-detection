
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier

from model.preprocessors.tokenizer import Tokenizer
from model.preprocessors.lemmatizer import lemmatize
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



data_directory = "data/1"
data = load_data(data_directory)

experiments_count = 0
results_path = "experiments_results/others/"

def get_experiment_file_name ():
    global experiments_count
    experiments_count += 1
    return f"{results_path}experiment{experiments_count}.png"

def show_groups_of_detectors_scores (all_detectors, training_data_weight, repeat):
    for detectors in all_detectors:
        manager = ExperimentManager(detectors, data)
        scores, rankings = manager.rank_detectors(training_data_weight, repeat)
        Metrics.plot_scores_and_rankings(scores, rankings, get_experiment_file_name())


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
    Metrics.plot_scores_and_rankings(scores, rankings, get_experiment_file_name())


def compare_featurizers (training_data_weight, repeat):
    all_detectors = [
        {
            "naives_bayes_cv": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "naives_bayes_tfidf": PlagiarismDetector(NaiveBayesClassifier(), TfidfFeaturizer(), Tokenizer())
        }, {
            "perceptron_cv": PlagiarismDetector(PerceptronClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "perceptron_tfidf": PlagiarismDetector(PerceptronClassifier(), TfidfFeaturizer(), Tokenizer())
        }, {
            "logistic_regression_cv": PlagiarismDetector(LogisticRegressionClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "logistic_regression_tfidf": PlagiarismDetector(LogisticRegressionClassifier(), TfidfFeaturizer(), Tokenizer())
        }, {
            "decision_tree_cv": PlagiarismDetector(DecisionTree(max_depth=1), CountVectorFeaturizer(), Tokenizer()),
            "decision_tree_tfidf": PlagiarismDetector(DecisionTree(max_depth=1), TfidfFeaturizer(), Tokenizer())
        }, {
            "ada_boost_stump_cv": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(), Tokenizer()),
            "ada_boost_stump_tfidf": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), TfidfFeaturizer(), Tokenizer())
        }, {
            "bagg_stump_cv": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorFeaturizer(), Tokenizer()),
            "bagg_stump_tfidf": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), TfidfFeaturizer(), Tokenizer())
        }
    ]
    show_groups_of_detectors_scores(all_detectors, training_data_weight, repeat)


def evaluate_ngrams_impact (training_data_weight, repeat):
    all_detectors = [
        {
            "naives_bayes_ngram_1_1": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "naives_bayes_ngram_1_2": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(ngram=[1,2]), Tokenizer()),
            "naives_bayes_ngram_2_2": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(ngram=[2,2]), Tokenizer()),
            "naives_bayes_ngram_1_3": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(ngram=[1,3]), Tokenizer()),
            "naives_bayes_ngram_3_3": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(ngram=[3,3]), Tokenizer()),
        }, {
            "ada_boost_ngram_1_1": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(), Tokenizer()),
            "ada_boost_ngram_1_2": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(ngram=[1,2]), Tokenizer()),
            "ada_boost_ngram_2_2": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(ngram=[2,2]), Tokenizer()),
            "ada_boost_ngram_1_3": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(ngram=[1,3]), Tokenizer()),
            "ada_boost_ngram_3_3": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(ngram=[3,3]), Tokenizer())
        }
    ]
    show_groups_of_detectors_scores(all_detectors, training_data_weight, repeat)


def evaluate_stop_words_impact (training_data_weight, repeat):
    all_detectors = [
        {
            "naives_bayes_delete_sw": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "naives_bayes_kept_sw": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer(stop_words=[]))
        }, {
            "perceptron_delete_sw": PlagiarismDetector(PerceptronClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "perceptron_kept_sw": PlagiarismDetector(PerceptronClassifier(), CountVectorFeaturizer(), Tokenizer(stop_words=[]))
        }, {
            "logistic_regression_delete_sw": PlagiarismDetector(LogisticRegressionClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "logistic_regression_kept_sw": PlagiarismDetector(LogisticRegressionClassifier(), CountVectorFeaturizer(), Tokenizer(stop_words=[]))
        }, {
            "decision_tree_delete_sw": PlagiarismDetector(DecisionTree(max_depth=1), CountVectorFeaturizer(), Tokenizer()),
            "decision_tree_kept_sw": PlagiarismDetector(DecisionTree(max_depth=1), CountVectorFeaturizer(), Tokenizer(stop_words=[]))
        }, {
            "ada_boost_stump_delete_sw": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(), Tokenizer()),
            "ada_boost_stump_kept_sw": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(), Tokenizer(stop_words=[]))
        }, {
            "bagg_stump_delete_sw": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorFeaturizer(), Tokenizer()),
            "bagg_stump_kept_sw": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorFeaturizer(), Tokenizer(stop_words=[]))
        }
    ]
    show_groups_of_detectors_scores(all_detectors, training_data_weight, repeat)


def evaluate_lemmatization_impact (training_data_weight, repeat):
    all_detectors = [
        {
            "naives_bayes": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "naives_bayes_lemmatization": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer(lemmatizer=lemmatize))
        }, {
            "perceptron": PlagiarismDetector(PerceptronClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "perceptron_lemmatization": PlagiarismDetector(PerceptronClassifier(), CountVectorFeaturizer(), Tokenizer(lemmatizer=lemmatize))
        }, {
            "logistic_regression": PlagiarismDetector(LogisticRegressionClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "logistic_regression_lemmatization": PlagiarismDetector(LogisticRegressionClassifier(), CountVectorFeaturizer(), Tokenizer(lemmatizer=lemmatize))
        }, {
            "decision_tree": PlagiarismDetector(DecisionTree(max_depth=1), CountVectorFeaturizer(), Tokenizer()),
            "decision_tree_lemmatization": PlagiarismDetector(DecisionTree(max_depth=1), CountVectorFeaturizer(), Tokenizer(lemmatizer=lemmatize))
        }, {
            "ada_boost_stump": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(), Tokenizer()),
            "ada_boost_stump_lemmatization": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(), Tokenizer(lemmatizer=lemmatize))
        }, {
            "bagg_stump": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorFeaturizer(), Tokenizer()),
            "bagg_stump_lemmatization": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorFeaturizer(), Tokenizer(lemmatizer=lemmatize))
        }
    ]
    show_groups_of_detectors_scores(all_detectors, training_data_weight, repeat)


def evaluate_customed_featurizer (training_data_weight, repeat):
    detectors = {
        "naives_bayes": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "naives_bayes_c": PlagiarismDetector(NaiveBayesClassifier(), CustomedFeaturizer(), Tokenizer())
    }
    manager = ExperimentManager(detectors, data)
    scores, rankings = manager.rank_detectors(training_data_weight, repeat)
    Metrics.plot_scores_and_rankings(scores, rankings, get_experiment_file_name())


def compare_with_scikit (training_data_weight, repeat):
    all_detectors = [
        {
            "naives_bayes": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "naives_bayes_scikit": PlagiarismDetector(MultinomialNB(), CountVectorizer(), Tokenizer())
        }, {
            "perceptron": PlagiarismDetector(PerceptronClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "perceptron_scikit": PlagiarismDetector(Perceptron(), CountVectorizer(), Tokenizer())
        }, {
            "logistic_regression": PlagiarismDetector(LogisticRegressionClassifier(), CountVectorFeaturizer(), Tokenizer()),
            "logistic_regression_scikit": PlagiarismDetector(LogisticRegression(), CountVectorizer(), Tokenizer())
        }, {
            "decision_tree": PlagiarismDetector(DecisionTree(max_depth=1), CountVectorFeaturizer(), Tokenizer()),
            "decision_tree_tfidf": PlagiarismDetector(DecisionTreeClassifier(max_depth=1), CountVectorizer(), Tokenizer())
        }, {
            "ada_boost_stump": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(), Tokenizer()),
            "ada_boost_stump_scikit": PlagiarismDetector(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorizer(), Tokenizer())
        }, {
            "bagg_stump": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorFeaturizer(), Tokenizer()),
            "bagg_stump_scikit": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorizer(), Tokenizer())
        }
    ]
    show_groups_of_detectors_scores(all_detectors, training_data_weight, repeat)


def compute_detectors_learning (training_data_weight, batch_size_increment):
    detectors = {
        "naives_bayes": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "perceptron": PlagiarismDetector(PerceptronClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "logistic_regression": PlagiarismDetector(LogisticRegressionClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "decision_tree": PlagiarismDetector(DecisionTree(max_depth=1), CountVectorFeaturizer(), Tokenizer()),
        "ada_boost_stump": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(), Tokenizer()),
        "bagg_stump": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorFeaturizer(), Tokenizer()),
    }
    manager = ExperimentManager(detectors, data)
    r = manager.evaluate_detectors_on_learning(training_data_weight, batch_size_increment)
    Metrics.plot_learning_curves(r, batch_size_increment, get_experiment_file_name())


def show_confusion_matrixs (training_data_weight):
    detectors = {
        "naives_bayes": PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "perceptron": PlagiarismDetector(PerceptronClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "logistic_regression": PlagiarismDetector(LogisticRegressionClassifier(), CountVectorFeaturizer(), Tokenizer()),
        "decision_tree": PlagiarismDetector(DecisionTree(max_depth=1), CountVectorFeaturizer(), Tokenizer()),
        "ada_boost_stump": PlagiarismDetector(AdaBoost(DecisionTreeClassifier(max_depth=1), labels_mapping_function=ada_boost_labels_mapping_function), CountVectorFeaturizer(), Tokenizer()),
        "bagg_stump": PlagiarismDetector(BaggingClassifier(DecisionTreeClassifier(max_depth=1)), CountVectorFeaturizer(), Tokenizer()),
    }
    manager = ExperimentManager(detectors, data)
    cms = manager.compute_accuracies_for_confusion_matrixs(training_data_weight)
    for detector in cms:
        y_true, y_pred = cms[detector]
        Metrics.plot_confusion_matrix(y_true, y_pred, detector, get_experiment_file_name())



if __name__ == "__main__":

    training_data_weight = 0.6
    repeat = 1000
    batch_size_increment = 10

    compute_detectors_ranks(training_data_weight, repeat)
    """compare_featurizers(training_data_weight, repeat)
    evaluate_ngrams_impact(training_data_weight, repeat)
    evaluate_stop_words_impact(training_data_weight, repeat)
    evaluate_lemmatization_impact(training_data_weight, repeat)
    evaluate_customed_featurizer(training_data_weight, repeat)
    compare_with_scikit(training_data_weight, repeat)
    compute_detectors_learning(training_data_weight, batch_size_increment)
    show_confusion_matrixs(training_data_weight)"""
