from featurizer import Featurizer


class TFIDFFeaturizer (Featurizer):

    def __init__ (self):
        super().__init__()
        self.idf = {}

    """
    def fit (self, documents):
        super().fit(documents)


    def transform (self, documents):
        return []
    """


if __name__ == "__main__":

    documents = [
        "Le chat noir est sur le toit.",
        "Le chat blanc dort sous la table.",
        "Le chien noir joue dans le jardin.",
        "La maison est grande et blanche.",
        "Sur le toit, le chat noir observe le jardin.",
        "Dans la maison, le chat blanc et le chien noir dorment ensemble.",
        "Le soleil brille sur le toit de la maison.",
        "Le chat noir court sur le toit tandis que le chien aboit."
    ]

    tfidf_f = TFIDFFeaturizer()
    tfidf_f.fit(documents)
    print(tfidf_f.vocabulary)
