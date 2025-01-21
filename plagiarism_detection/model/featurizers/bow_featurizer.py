from featurizer import Featurizer


class BoWFeaturizer (Featurizer):

    def __init__ (self):
        super().__init__()

    def transform (self, documents):
        X = []
        for document in documents:
            vector = [0] * len(self.vocabulary)
            for word in document.split():
                if word in self.vocabulary:
                    vector[self.vocabulary[word]] += 1
            X.append(vector)
        return X



if __name__ == "__main__":

    documents = ["chat noir", "chat blanc", "noir et blanc"]
    bow_f = BoWFeaturizer()

    bow_f.fit(documents)
    print(bow_f.transform(documents))