
import numpy as np
from collections import Counter
from typing import List
from model.tokenizers.tokenizer import Tokenizer


class CountVectorFeaturizer:
    def __init__(self):
        self.vocabulary = {} 
        self.tokenizer = Tokenizer() 
        self.tokenized_docs = [] 

    def fit(self, documents: List[str]) -> None:
        token_counter = Counter()  
        self.tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents] 
        
        for tokens in self.tokenized_docs:
            ngrams = Tokenizer.generate_ngrams(tokens=tokens, ngram=[1,1])
            token_counter.update(set(ngrams))
        
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(token_counter.most_common())}

    def transform(self, documents: List[str]) -> np.ndarray:
        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]  
        count_matrix = np.zeros((len(documents), len(self.vocabulary)), dtype=int)

        for i, tokens in enumerate(tokenized_docs):
            ngrams = Tokenizer.generate_ngrams(tokens=tokens, ngram=[1, 1])
            for token in ngrams:
                idx = self.vocabulary.get(token)
                if idx is not None:
                    count_matrix[i, idx] += 1
        
        return count_matrix

