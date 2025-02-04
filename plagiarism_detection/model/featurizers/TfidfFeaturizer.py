#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from typing import List
from collections import Counter
from Tokenizer import Tokenizer


# In[2]:


# Term freaquency: frequency of all words within a document
# Inverse document Frequency: inverse of how often a word appears accross al documents 
class TfidfFeaturizer:
    def __init__(self):
        self.vocabulary = {}  
        self.tokenizer = Tokenizer()  
        #self.tokenized_docs = []  
        self.idf_values = {} #stores the calculated idf values

    def fit(self, documents: List[str]) -> None:
        nb_doc = len(documents) 
        token_counter = Counter()  
        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]
        
        for tokens in tokenized_docs:
            token_counter.update(set(tokens)) #we only count each occurence of a token in a doc once, in other terms the number of times a token appears in all docs
            
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(token_counter.most_common())}
        
        #print("vocabulary", self.vocabulary)
        #Calculate the idf values of each token, log of the number of docs devided by the number of docs that contain the term (+1 to avoid 0 values)
        self.idf_values = {
            token: np.log((1 + nb_doc) / (1 + doc_freq)) + 1
            for token, doc_freq in token_counter.items()
        }

    def transform(self, documents: List[str]) -> np.ndarray:
        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]  # Tokenize input documents
        tfidf_matrix = np.zeros((len(documents), len(self.vocabulary)))

        #for each token:term of each document we calculate tf*idf
        for i, tokens in enumerate(tokenized_docs):
            term_counts = Counter(tokens) #transformation to a counter
            #print(term_counts)
            total_terms = len(tokens)
            for token, count in term_counts.items(): 
                tf = count / total_terms  # Term frequency
                idf = self.idf_values.get(token, 0)  # IDF value (default to 0 if missing)
                if token in self.vocabulary:
                     tfidf_matrix[i, self.vocabulary[token]] = tf * idf

        return tfidf_matrix


# In[ ]:


"""folder_path = "../data-plagiarism"
documents = []

featurizer = TfidfFeaturizer()

for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        try:
            content = Tokenizer.read_file(file_path)  # Use the Tokenizer class's file reading method
            documents.append(content)
        except ValueError as e:
            print(f"Error reading {file_name}: {e}")

# Fit the featurizer and transform the documents into a count matrix
featurizer.fit(documents)
count_matrix = featurizer.transform(documents)

# Print the results
#print("Vocabulary:", featurizer.vocabulary)
print("Count Matrix:\n", count_matrix)"""


# In[ ]:




