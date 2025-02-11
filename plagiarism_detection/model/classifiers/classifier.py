from abc import ABC, abstractmethod


class Classifier (ABC):

    def __init__ (self, classes_count=0):
        self.class_priors = {}

    """
    @abstractmethod
    def train (self, X, y):
        pass

    @abstractmethod
    def predict (self, X):
        pass
    """
    
    def evaluate (self, X, y):
        y_computed = self.predict(X)
        corrects = sum(1 for computed, expected in zip(y_computed, y) if computed == expected)
        return corrects / len(y)

    def init_class_priors (self, classes_count):
        default_prob = 1 / classes_count
        self.class_priors = {classs: default_prob for classs in range(classes_count)}