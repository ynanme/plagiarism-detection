import numpy as np
from abc import ABC, abstractmethod

from model.tokenizers.tokenizer import Tokenizer


class Featurizer (ABC):

    def __init__ (self, tokenizer=Tokenizer()):
        self.vocabulary = {}
        self.tokenizer = tokenizer

    """
    def fit (self, documents):
        for doc in documents:
            for word in doc.split():
                if not word in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)
    """

    @abstractmethod
    def transform (self, documents):
        pass

    def fit_transform (self, documents):
        self.fit(documents)
        return self.transform(documents)


class CustomedFeaturizer (Featurizer):

    def __init__ (self,  tokenizer=Tokenizer()):
        super().__init__()

    def fit (self, documents):
        pass

    def transform (self, documents):
        return np.repeat(np.array([len(document) for document in documents])[:,np.newaxis], 3, axis=1)