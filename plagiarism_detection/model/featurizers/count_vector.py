import numpy as np
from collections import Counter
from typing import List

from model.tokenizers.tokenizer import Tokenizer
from model.featurizers.featurizer import Featurizer


class CountVectorFeaturizer (Featurizer):

    def __init__(self, tokenizer=Tokenizer()):
        super().__init__(tokenizer)
        self.tokenized_docs = [] #stores tokenized docs to optimise space and code 

    def fit(self, documents: List[str]) -> None:
        token_counter = Counter()  # Counter to keep track of token frequencies
        self.tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents] #Tokenize and store all docs
        
        # Update the counter with tokens from each document (unique + count)
        for tokens in self.tokenized_docs:
            token_counter.update(tokens)
        
        # Map each token to a unique index based on frequency
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(token_counter.most_common())}

    def transform(self, documents: List[str]) -> np.ndarray:

        # Initialize a matrix for the count vectors
        count_matrix = np.zeros((len(documents), len(self.vocabulary)), dtype=int)

        # Fill the matrix with token frequencies based on the vocabulary
        for i, tokens in enumerate(self.tokenized_docs): #for each token of each document, if in vocab then we inc that cell in the matrix
            for token in tokens:
                idx = self.vocabulary.get(token)
                if idx is not None:
                    count_matrix[i, idx] += 1
        
        return count_matrix



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

    cv_f = CountVectorFeaturizer()
    
    cv_f.fit(documents)
    print(cv_f.vocabulary)

    for line in cv_f.transform(documents[1::3]):
        print(line)