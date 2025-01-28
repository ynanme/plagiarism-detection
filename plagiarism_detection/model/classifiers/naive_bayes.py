import numpy as np
from typing import List, Dict

from model.classifiers.classifier import Classifier


class NaiveBayesClassifier (Classifier):
    def __init__(self):
        self.class_priors = {} #{plagiat, non_plagiat}
        self.word_probabilities = {} #word_likelihoods

    def fit(self, filtered_labels, featurizer, matrix):
        # Set class priors
        
        self.class_priors = compute_class_priors(filtered_labels)
        
        # Set word likelihoods
        self.word_probabilities = compute_word_probabilities(filtered_labels, featurizer, matrix)
        """if isinstance(featurizer, CountVectorFeaturizer):
            self.word_probabilities = compute_word_probabilities(filtered_labels, featurizer, matrix)
        else:
            self.word_probabilities = compute_word_probabilities_tfidf(filtered_labels, featurizer, matrix)"""
    def predict(self, tokens):
        #initialize scores for all classes(plagiat + non_plagiat)
        scores = {cls: np.log(self.class_priors[cls]) for cls in self.class_priors if self.class_priors[cls] > 0}
        #for each token we calculate log of likelihoods for each class
        for token in tokens:
            if token in self.word_probabilities:
                for cls in scores:
                    scores[cls] += np.log(self.word_probabilities[token][cls])
                    
        # Return class with highest score
        return max(scores, key=scores.get)



def compute_class_priors(filtered_labels):
    total_files = len(filtered_labels)
    plagiat_files = sum(1 for v in filtered_labels.values() if v == "plagiat") #number or plagiat files
    #print(plagiat_files, total_files)
    return {
        "plagiat": plagiat_files / total_files, #probability of a file been plagiat P(plagiat)
        "non_plagiat": 1 - (plagiat_files / total_files) #probability of a file been non plagiat P(non_plagiat)
    }

def compute_word_probabilities(filtered_labels: Dict[str, str], featurizer, matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    
    totals = np.zeros((len(featurizer.vocabulary), 2)) #matrix that's gonna contain the total occurence of a word in all docs, and in plagiat docs
    for file_name, row in zip(filtered_labels.keys(), matrix):
        for word, index in featurizer.vocabulary.items():
            totals[index, 0] += row[index] # Add the word's count from the feature matrix to the total counts.
            if filtered_labels[file_name] == 'plagiat':
                totals[index, 1] += row[index] #if the file is plagiat then add the count to the second column

    """print(matrix)
    print(filtered_labels)
    print([filtered_labels[file_name] == 'plagiat' for file_name in filtered_labels.keys()])
    print([filtered_labels[file_name] == 'non_plagiat' for file_name in filtered_labels.keys()])"""
    total_plagiat_words = matrix[[filtered_labels[file_name] == 'plagiat' for file_name in filtered_labels.keys()]].sum().sum()
    total_non_plagiat_words = matrix[[filtered_labels[file_name] == 'non_plagiat' for file_name in filtered_labels.keys()]].sum().sum()

    word_totals = {}
    for word, index in featurizer.vocabulary.items():
        total = totals[index, 0]
        total_plagiat = totals[index, 1]
        #the likelihoods of each word == P(word/class) 
        word_totals[word] = {
            "plagiat": (total_plagiat + 1) / total_plagiat_words if total > 0 and total_plagiat_words > 0 else 0,
            "non_plagiat": (total - total_plagiat + 1) / total_non_plagiat_words if total_non_plagiat_words > 0 else 0
        }

    return word_totals

def compute_word_probabilities_tfidf(filtered_labels: Dict[str, str], featurizer, matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    totals = np.zeros((len(featurizer.vocabulary), 2))  # TF-IDF sums for each class

    for file_name, row in zip(filtered_labels.keys(), matrix):
        for word, index in featurizer.vocabulary.items():
            if filtered_labels[file_name] == 'plagiat':
                totals[index, 1] += row[index]
            else:
                totals[index, 0] += row[index]

    total_plagiat_tfidf = totals[:, 1].sum()
    total_non_plagiat_tfidf = totals[:, 0].sum()

    word_totals = {}
    for word, index in featurizer.vocabulary.items():
        word_totals[word] = {
            "plagiat": (totals[index, 1] + 1) / (total_plagiat_tfidf + 1),  # Smooth with Laplace
            "non_plagiat": (totals[index, 0] + 1) / (total_non_plagiat_tfidf + 1),
        }

    return word_totals