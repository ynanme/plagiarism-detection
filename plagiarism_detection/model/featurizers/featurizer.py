from abc import ABC, abstractmethod


class Featurizer (ABC):

    def __init__ (self):
        self.vocabulary = {}

    def fit (self, documents):
        for doc in documents:
            for word in doc.split():
                if not word in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)

    @abstractmethod
    def transform (self, documents):
        pass

    def fit_transform (self, documents):
        self.fit(documents)
        return self.transform(documents)