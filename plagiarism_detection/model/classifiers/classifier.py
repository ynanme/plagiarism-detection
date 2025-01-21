from abc import ABC, abstractmethod


class Classifier (ABC):

    @abstractmethod
    def train (self, X, y):
        pass

    @abstractmethod
    def predict (self, X):
        pass

    def evaluate (self, X, y):
        y_computed = self.predict(X)
        corrects = sum(1 for computed, expected in zip(y_computed, y) if computed == expected)
        return corrects / len(y)