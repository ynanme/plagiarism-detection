import numpy as np

from abc import ABC, abstractmethod


class Classifier (ABC):

    def __init__ (self, classes_count=0):
        self.classes = np.array([])
        self.class_priors = np.array([])

    
    def fit (self, features_matrix, labels):
        self.classes = np.unique(labels)

    """
    @abstractmethod
    def predict (self, X):
        pass
    """
    
    def evaluate (self, X, y):
        y_computed = self.predict(X)
        corrects = sum(1 for computed, expected in zip(y_computed, y) if computed == expected)
        return corrects / len(y)

    #def score (self, features_matrix, labels, arg=np.mean):


    def init_class_priors (self, classes_count):
        default_prob = 1 / classes_count
        self.class_priors = {classs: default_prob for classs in range(classes_count)}

    #def update_class_priors (self, y):
        