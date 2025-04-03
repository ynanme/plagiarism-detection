import numpy as np
from abc import ABC, abstractmethod

from model.preprocessors.tokenizer import Tokenizer


class Featurizer (ABC):

    def __init__ (self, tokenizer=Tokenizer(), ngram=[1, 1]):
        self.vocabulary = {}
        self.tokenizer = tokenizer
        self.ngram = ngram


    def set_tokenizer (self, new_tokenizer):
        self.tokenizer = new_tokenizer

    @abstractmethod
    def fit (self, documents):
        pass

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
        return np.repeat(np.array([len(document) for document in documents])[:,np.newaxis], 30, axis=1)