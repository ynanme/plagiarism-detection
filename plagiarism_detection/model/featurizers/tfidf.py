import numpy as np
from typing import List
from collections import Counter

from model.featurizers.featurizer import Featurizer
from model.tokenizers.tokenizer import Tokenizer


# Term freaquency: frequency of all words within a document
# Inverse document Frequency: inverse of how often a word appears accross al documents 
class TfidfFeaturizer (Featurizer):

    def __init__(self, tokenizer=Tokenizer()):
        super().__init__(tokenizer)
        self.tokenized_docs = []  
        self.idf_values = {} #stores the calculated idf values

    def fit(self, documents: List[str]) -> None:
        nb_doc = len(documents) 
        token_counter = Counter()  
        self.tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]
        
        for tokens in self.tokenized_docs:
            token_counter.update(set(tokens)) #we only count each occurence of a token in a doc once, in other terms the number of times a token appears in all docs
            
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(token_counter.most_common())}
        
        #print("vocabulary", self.vocabulary)
        #Calculate the idf values of each token, log of the number of docs devided by the number of docs that contain the term (+1 to avoid 0 values)
        self.idf_values = {
            token: np.log((1 + nb_doc) / (1 + doc_freq)) + 1
            for token, doc_freq in token_counter.items()
        }

    def transform(self, documents: List[str]) -> np.ndarray:
        tfidf_matrix = np.zeros((len(documents), len(self.vocabulary)))

        #for each token:term of each document we calculate tf*idf
        for i, tokens in enumerate(self.tokenized_docs):
            term_counts = Counter(tokens) #transformation to a counter
            #print(term_counts)
            total_terms = len(tokens)
            for token, count in term_counts.items(): 
                tf = count / total_terms  # Term frequency
                idf = self.idf_values.get(token, 0)  # IDF value (default to 0 if missing)
                tfidf_matrix[i, self.vocabulary[token]] = tf * idf

        return tfidf_matrix



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


