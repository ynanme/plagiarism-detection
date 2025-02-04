#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import numpy as np
from collections import Counter
from typing import List
#import import_ipynb
#from Tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


class CountVectorFeaturizer:
    def __init__(self):
        self.vocabulary = {}  #stores vocabulary => index : token
        self.tokenizer = Tokenizer() 
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
        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]  # Tokenize input documents
        # Initialize a matrix for the count vectors
        count_matrix = np.zeros((len(documents), len(self.vocabulary)), dtype=int)

        # Fill the matrix with token frequencies based on the vocabulary
        for i, tokens in enumerate(self.tokenized_docs): #for each token of each document, if in vocab then we inc that cell in the matrix
            for token in tokens:
                idx = self.vocabulary.get(token)
                if idx is not None:
                    count_matrix[i, idx] += 1
        
        return count_matrix


# In[3]:


"""folder_path = "../data-plagiarism"
documents = []

featurizer = CountVectorFeaturizer()

for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        try:
            content = Tokenizer.read_file(file_path)  # Use the Tokenizer class's file reading method
            documents.append(content)
        except ValueError as e:
            print(f"Error reading {file_name}: {e}")

# Fit the featurizer and transform the documents into a count count_matrix
featurizer.fit(documents)
count_count_matrix = featurizer.transform(documents)

# Print the results
#print("Vocabulary:", featurizer.vocabulary)
print("Count count_matrix:\n", count_count_matrix)"""


# In[12]:


a = """Inheritance is a basic concept of Object-Oriented Programming where
the basic idea is to create new classes that add extra detail to
existing classes. This is done by allowing the new classes to reuse
the methods and variables of the existing classes and new methods and
classes are added to specialise the new class. Inheritance models the
“is-kind-of” relationship between entities (or objects), for example,
postgraduates and undergraduates are both kinds of student. This kind
of relationship can be visualised as a tree structure, where ‘student’
would be the more general root node and both ‘postgraduate’ and
‘undergraduate’ would be more specialised extensions of the ‘student’
node (or the child nodes). In this relationship ‘student’ would be
known as the superclass or parent class whereas, ‘postgraduate’ would
be known as the subclass or child class because the ‘postgraduate’
class extends the ‘student’ class.

Inheritance can occur on several layers, where if visualised would
display a larger tree structure. For example, we could further extend
the ‘postgraduate’ node by adding two extra extended classes to it
called, ‘MSc Student’ and ‘PhD Student’ as both these types of student
are kinds of postgraduate student. This would mean that both the ‘MSc
Student’ and ‘PhD Student’ classes would inherit methods and variables
from both the ‘postgraduate’ and ‘student classes’."""

CountVectorizer().build_tokenizer()


# In[ ]:




