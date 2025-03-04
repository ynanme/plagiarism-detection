import numpy as np
from typing import Dict


class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}  # P(Class)
        self.word_probabilities = {}  # P(Word | Class)

    def fit(self, filtered_labels: Dict[str, int], featurizer, matrix: np.ndarray):

        self.class_priors = self.compute_class_priors(filtered_labels)
        self.word_probabilities = self.compute_word_probabilities(filtered_labels, featurizer, matrix)

    def predict(self, tokens):

        scores = {cls: np.log(self.class_priors[cls]) for cls in self.class_priors}

        # Compute log-likelihood for each class
        for token in tokens:
            if token in self.word_probabilities:
                for cls in scores:
                    prob = self.word_probabilities[token].get(cls, 1 / (sum(self.word_probabilities[word][cls] for word in self.word_probabilities) + len(self.word_probabilities)))
                    scores[cls] += np.log(prob)

        return max(scores, key=scores.get)

    @staticmethod
    def compute_class_priors(filtered_labels: np.ndarray) -> Dict[str, float]:

        total_files = len(filtered_labels)
        plagiat_files = np.sum(filtered_labels) 
        non_plagiat_files = total_files - plagiat_files  

        return {
            "plagiat": plagiat_files / total_files if total_files > 0 else 0,
            "non_plagiat": non_plagiat_files / total_files if total_files > 0 else 0
        }


    @staticmethod
    def compute_word_probabilities(filtered_labels: np.ndarray, featurizer, matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
        vocabulary_size = len(featurizer.vocabulary)
        totals = np.zeros((vocabulary_size, 2))  
        for i, row in enumerate(matrix):
            for word, index in featurizer.vocabulary.items():
                totals[index, 0] += row[index]  
                if filtered_labels[i]  == 1:  
                    totals[index, 1] += row[index]

        # Compute total word occurrences per class
        total_plagiat_words = totals[:, 1].sum()
        total_non_plagiat_words = totals[:, 0].sum() - total_plagiat_words

        word_totals = {}
        for word, index in featurizer.vocabulary.items():
            total = totals[index, 0]
            total_plagiat = totals[index, 1]
            
            # Apply Laplace Smoothing to avoid zero probabilities
            word_totals[word] = {
                "plagiat": (total_plagiat + 1) / (total_plagiat_words + vocabulary_size),
                "non_plagiat": (total - total_plagiat + 1) / (total_non_plagiat_words + vocabulary_size)
            }

        return word_totals

