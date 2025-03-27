import numpy as np

from abc import ABC, abstractmethod


class Classifier (ABC):

    def __init__ (self, classes=np.array([])):
        self.classes = classes

    def fit (self, features_matrix, labels):
        self.classes = np.unique(labels)

    @abstractmethod
    def predict (self, features_matrix):
        pass
    
