#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from tokenizer import Tokenizer
from CountVectorFeaturizer import CountVectorFeaturizer
from TfidfFeaturizer import TfidfFeaturizer
from NaiveBayesClassifier import NaiveBayesClassifier
from LogisticRegressionClassifier import LogisticRegressionClassifier 


# In[4]:

class PerceptronClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, matrix, binary_labels):
        nb_docs, nb_vocab = matrix.shape
        self.weights = np.zeros(nb_vocab)  # Initialize weights to zero

        for epoch in range(self.epochs):
            #print(f"Epoch {epoch} weight {self.weights} bias {self.bias}")
            for i in range(nb_docs):
                z = np.dot(matrix[i], self.weights) + self.bias
                if - binary_labels[i] * z > 0:
                    # Update weights and bias
                    self.weights += self.learning_rate * binary_labels[i] * matrix[i]
                    self.bias += self.learning_rate * binary_labels[i]
                    

    def predict(self, matrix):
        z = np.dot(matrix, self.weights) + self.bias
        #print("linear_output", np.sign(linear_output))
        predictions = np.sign(z)
        return predictions


# In[5]:


"""folder_path = "../data-plagiarism"

# Load labels
filtered_labels = load_labels(folder_path, "file_information.csv")
test_labels = load_labels(folder_path, "test_info.csv")
train_labels = {file: label for file, label in filtered_labels.items() if file not in test_labels}
documents = [Tokenizer.read_file(os.path.join(folder_path, file)) for file in train_labels.keys()]#read all documents

# Test both featurizers
for featurizer_class in [CountVectorFeaturizer, TfidfFeaturizer]:
    featurizer = featurizer_class()
    featurizer.fit(documents)
    matrix = featurizer.transform(documents)
    #print("matrix", matrix)
    #word_probabilities = compute_word_probabilities(filtered_labels, featurizer, matrix)
    classifier = PerceptronClassifier()
    binary_labels = [1 if label == "plagiat" else -1 for label in train_labels.values()]
    #print("binary labels", binary_labels)
    #print("matrix", filtered_labels)
    classifier.fit(matrix, binary_labels)
    
    # Classify files
    results = []
    for file_name, feature_vector in zip(test_labels.keys(), matrix):
        prediction = classifier.predict(feature_vector.reshape(1, -1))[0]
        results.append({
            "File Name": file_name,
            "Expected Label": filtered_labels[file_name],
            "Predicted Label": "plagiat" if prediction == 1 else "non_plagiat"
        })

    df_results = pd.DataFrame(results)
    df_results["Mismatch"] = df_results["Expected Label"] != df_results["Predicted Label"]

    print(f"Results for {featurizer_class.__name__}:")
    #print(df_results.to_string(index=False))

    mismatches = df_results[df_results["Mismatch"]]
    print(f"Mismatches for {featurizer_class.__name__}:", len(mismatches))
    #print(mismatches.to_string(index=False))
    
"""


# In[ ]:




