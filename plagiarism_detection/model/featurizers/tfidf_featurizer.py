from featurizer import Featurizer


class TFIDFFeaturizer (Featurizer):

    def __init__ (self):
        super().__init__()
        self.idf = {}

    def fit (self, documents):
        super().fit(documents)
        num_docs = len(documents)
        word_doc_count = Counter()

        tokenized_docs = [doc.split() for doc in documents]
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            word_doc_count.update(unique_tokens)

        self.idf = {
            word: math.log(num_docs / (1 + count)) for word, count in word_doc_count.items()
        }


    def transform (self, documents):
        num_words = len(self.vocabulary)
        matrix = []

        for doc in documents:
            tf = Counter(doc.split())
            total_words = len(doc.split())
            row = [0] * num_words

            for word, count in tf.items():
                if word in self.vocabulary:
                    tf_value = count / total_words
                    idf_value = self.idf[word]
                    row[self.vocabulary[word]] = tf_value * idf_value

            matrix.append(row)
        return matrix



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


