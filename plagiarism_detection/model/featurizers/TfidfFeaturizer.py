import numpy as np
from typing import List
from collections import Counter
from model.tokenizers.tokenizer import Tokenizer


# Term freaquency: frequency of all words within a document
# Inverse document Frequency: inverse of how often a word appears accross al documents 
class TfidfFeaturizer:
    def __init__(self, ngram=[1, 1]):
        self.vocabulary = {}  
        self.tokenizer = Tokenizer()  
        #self.tokenized_docs = []  
        self.idf_values = {}
        self.ngram = ngram

    def fit(self, documents: List[str]) -> None:
        nb_doc = len(documents) 
        token_counter = Counter()  

        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]
        
        for tokens in tokenized_docs:
            ngrams = Tokenizer.generate_ngrams(tokens=tokens, ngram=self.ngram)
            token_counter.update(set(ngrams)) #we only count each occurence of a token in a doc once, in other terms the number of times a token appears in all docs
        
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(token_counter.most_common())}

        #Calculate the idf values of each token, log of the number of docs devided by the number of docs that contain the term (+1 to avoid 0 values)
        self.idf_values = {
            token: np.log((nb_doc) / (1 + doc_freq)) + 1
            for token, doc_freq in token_counter.items()
        }

    def transform(self, documents: List[str]) -> np.ndarray:
        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]  # Tokenize input documents
        tfidf_matrix = np.zeros((len(documents), len(self.vocabulary)))

        #for each token:term of each document we calculate tf*idf
        for i, tokens in enumerate(tokenized_docs):
            ngrams = Tokenizer.generate_ngrams(tokens=tokens, ngram=self.ngram)
            term_counts = Counter(ngrams) #transformation to a counter
            total_terms = len(ngrams)

            for token, count in term_counts.items(): 
                tf = count / total_terms  # Term frequency
                idf = self.idf_values.get(token, 0)  
                if token in self.vocabulary:
                     tfidf_matrix[i, self.vocabulary[token]] = tf * idf
        return tfidf_matrix

