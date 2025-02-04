#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
from typing import List, Dict

from tokenizer import Tokenizer

from CountVectorFeaturizer import CountVectorFeaturizer
from TfidfFeaturizer import TfidfFeaturizer

from NaiveBayesClassifier import NaiveBayesClassifier
from LogisticRegressionClassifier import LogisticRegressionClassifier 
from PerceptronClassifier import PerceptronClassifier


# In[ ]:


def map_value(value: str) -> str:
    return 'plagiat' if value in ['heavy', 'cut'] else 'non_plagiat'

def load_labels(folder_path: str, file_name: str) -> Dict[str, str]:
    expected_results = pd.read_csv(os.path.join(folder_path, file_name), sep=",")
    labels = dict(zip(expected_results["File"], expected_results['Category']))
    return {k: map_value(v) for k, v in sorted(labels.items())}


# In[ ]:


if __name__ == "__main__":
    folder_path = "../data-plagiarism"

    # Load labels
    filtered_labels = load_labels(folder_path, "file_information.csv")
    test_labels = load_labels(folder_path, "test_info.csv")
    train_labels = {file: label for file, label in filtered_labels.items() if file not in test_labels}
    #print(len(train_labels))
    documents = [Tokenizer.read_file(os.path.join(folder_path, file)) for file in train_labels.keys()]  # Read all documents

    for featurizer_class in [CountVectorFeaturizer, TfidfFeaturizer]:
        featurizer = featurizer_class()
        featurizer.fit(documents)
        matrix = featurizer.transform(documents)

        for classifier_class in [NaiveBayesClassifier, PerceptronClassifier, LogisticRegressionClassifier]:
            classifier = classifier_class()

            # Logistic Regression and Perceptron require binary labels (0/1 or -1/1)
            if classifier_class in [LogisticRegressionClassifier, PerceptronClassifier]:
                binary_labels = (
                    np.array([1 if label == "plagiat" else 0 for label in train_labels.values()])
                    if classifier_class in [LogisticRegressionClassifier]
                    else [1 if label == "plagiat" else -1 for label in train_labels.values()]
                )
                classifier.fit(matrix, binary_labels)
            else:
                classifier.fit(train_labels, featurizer, matrix)

            # Classify files
            results = []
            for file_name, feature_vector in zip(test_labels.keys(), matrix):
                if classifier_class in [LogisticRegressionClassifier, PerceptronClassifier]:
                    # Predict with feature vector
                    prediction = classifier.predict([feature_vector])[0] # or can use .reshape(1, -1)
                    predicted_label = "plagiat" if prediction == 1 else "non_plagiat"
                else:
                    # Tokenize and predict for Naive Bayes
                    content = Tokenizer.read_file(os.path.join(folder_path, file_name))
                    tokens = featurizer.tokenizer.tokenize(content)
                    predicted_label = classifier.predict(tokens)

                results.append({
                    "File Name": file_name,
                    "Expected Label": filtered_labels[file_name],
                    "Predicted Label": predicted_label
                })

            # Create results DataFrame
            df_results = pd.DataFrame(results)
            df_results["Mismatch"] = df_results["Expected Label"] != df_results["Predicted Label"]

            # Print results
            #print(f"Results for {featurizer_class.__name__} with {classifier_class.__name__}:")
            #print(df_results.to_string(index=False))

            mismatches = df_results[df_results["Mismatch"]]
            print(f"Mismatches for {featurizer_class.__name__} with {classifier_class.__name__}: {len(mismatches)}\n")
            #print(mismatches.to_string(index=False))

            
#TODO: use binary labels with naiveBayes


# In[ ]:





# In[ ]:




