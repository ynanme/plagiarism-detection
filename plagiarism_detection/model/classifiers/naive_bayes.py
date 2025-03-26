import numpy as np
from typing import List, Dict

from model.classifiers.classifier import Classifier


class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = None
        self.word_probabilities = None
        self.vocab_size = None

    def fit(self, matrix, filtered_labels):
        self.vocab_size = matrix.shape[1]
        self.class_priors = self.compute_class_priors(matrix, filtered_labels)
        self.word_probabilities = self.compute_word_probabilities(matrix, filtered_labels)

    def predict(self, matrix):
        if isinstance(matrix, list):
            matrix = np.array([matrix])

        total_files = matrix.shape[0]
        nb_classes = len(self.class_priors)

        log_probs = np.zeros((total_files, nb_classes))

        for cls in range(nb_classes):
            log_probs[:, cls] = np.log(self.class_priors[cls])

            for i in range(total_files):
                log_probs[i, cls] += np.sum(matrix[i] * np.log(self.word_probabilities[cls]))

        return np.argmax(log_probs, axis=1)

    @staticmethod
    def compute_class_priors(matrix, filtered_labels):
        total_files = matrix.shape[0]
        num_classes = len(np.unique(filtered_labels))

        class_priors = np.zeros(num_classes)
        for cls in range(num_classes):
            class_priors[cls] = np.sum(filtered_labels == cls) / total_files

        return class_priors
    
    @staticmethod
    def compute_word_probabilities(matrix, filtered_labels):
        vocab_size = matrix.shape[1]
        word_probabilities = np.zeros((len(np.unique(filtered_labels)), vocab_size))

        for cls in range(len(np.unique(filtered_labels))):
            word_cls = matrix[filtered_labels == cls] 
            word_counts = np.sum(word_cls, axis=0)
            total_count = np.sum(word_counts) 
            word_probabilities[cls] = (word_counts + 1) / (total_count + vocab_size)

        return word_probabilities


    def __str__ (self):
        return "NAIVE BAYES"




class NaiveBayesClassifier2 (Classifier):

    def __init__ (self):
        super().__init__()
        self.features_priors = np.array([])
        self.features_count = 0
        self.class_priors_weight = 0

    def fit (self, X, y):
        super().fit(X, y)
        self.class_priors_weight, self.features_count = X.shape
        self.update_class_priors(y)
        self.update_features_probs(X, y)

    def update_class_priors (self, y):
        self.class_priors = np.zeros(len(self.classes))
        for i, classs in enumerate(self.classes):
            self.class_priors[i] = np.sum(y == classs) / len(y)

    def update_features_probs (self, X, y):
        #indexes = np.arange(len(y))
        features_counts_for_each_class = np.array([np.sum(X[y==classs], axis=0) for classs in self.classes]).T
        features_counts_for_all_classes = np.sum(X, axis=0).T
        #features_counts = np.concatenate((features_counts_for_each_class, features_counts_for_all_classes[:, np.newaxis]), axis=1)
        self.features_prob = features_counts_for_each_class / features_counts_for_all_classes[:, np.newaxis] + 1

    def predict (self, features_matrix):
        predictions = []
        for i in range(len(self.classes)):
            prediction = np.log(self.class_priors[i])
            prediction += np.sum(features_matrix * np.log(self.features_prob.T[i]), axis=1)
            predictions.append(prediction)
        print(predictions)
        return self.classes[np.argmax(predictions, axis=0)]


    def __str__ (self):
        return "NAIVE BAYES"

