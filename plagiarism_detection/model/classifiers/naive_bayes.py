import numpy as np
from typing import List, Dict

from model.classifiers.classifier import Classifier


class NaiveBayesClassifier (Classifier):

    def __init__ (self):
        super().__init__()
        self.features_priors = np.array([])
        self.features_count = 0
        self.class_priors_weight = 0

    def fit (self, X, y):
        super().fit(X, y)
        self.class_priors_weight, self.features_count = X.shape
        self.update_class_priors(y)
        self.update_features_probs(X, y)

    def update_class_priors (self, y):
        self.class_priors = np.zeros(len(self.classes))
        for i, classs in enumerate(self.classes):
            self.class_priors[i] = np.sum(y == classs) / len(y)

    def update_features_probs (self, X, y):
        #indexes = np.arange(len(y))
        features_counts_for_each_class = np.array([np.sum(X[y==classs], axis=0) for classs in self.classes]).T
        features_counts_for_all_classes = np.sum(X, axis=0).T
        #features_counts = np.concatenate((features_counts_for_each_class, features_counts_for_all_classes[:, np.newaxis]), axis=1)
        self.features_prob = features_counts_for_each_class / features_counts_for_all_classes[:, np.newaxis] + 1

    def predict (self, features_matrix):
        predictions = []
        for i in range(len(self.classes)):
            prediction = np.log(self.class_priors[i])
            prediction += np.sum(features_matrix * np.log(self.features_prob.T[i]), axis=1)
            predictions.append(prediction)
        print(predictions)
        return self.classes[np.argmax(predictions, axis=0)]


    def __str__ (self):
        return "NAIVE BAYES"


class NaiveBayesClassifier2 (Classifier):
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
        scores = {cls: np.log(self.class_priors[cls]) for cls in self.class_priors}

        #for each token we calculate log of likelihoods for each class
        for token in tokens:
            if token in self.word_probabilities:
                for cls in scores:
                    scores[cls] += np.log(self.word_probabilities[token][cls])
                    
        # Return class with highest score
        return max(scores, key=scores.get)



def compute_class_priors(filtered_labels):
    print("wtf", filtered_labels)
    total_files = len(filtered_labels)
    plagiat_files = np.sum(filtered_labels, where=1)
    #print("plagiat_files", plagiat_files) 
    return {
        1: plagiat_files / total_files, #probability of a file been plagiat P(plagiat)
        0: 1 - (plagiat_files / total_files) #probability of a file been non plagiat P(non_plagiat)
    }

def compute_word_probabilities(filtered_labels, featurizer, matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    print(filtered_labels)
    totals = np.zeros((len(featurizer.vocabulary), 2)) #matrix that's gonna contain the total occurence of a word in all docs, and in plagiat docs
    print("the zip", list(zip(filtered_labels, matrix)))
    for file_name, row in zip(filtered_labels, matrix):
        for word, index in featurizer.vocabulary.items():
            totals[index, 0] += row[index] # Add the word's count from the feature matrix to the total counts.
            print("here", file_name, len(filtered_labels))
            #if filtered_labels[file_name] == 1:
            if file_name == 1:
                totals[index, 1] += row[index] #if the file is plagiat then add the count to the second column

    #total_plagiat_words = matrix[[filtered_labels[file_name] == 1 for file_name in filtered_labels]].sum().sum()
    #total_non_plagiat_words = matrix[[filtered_labels[file_name] == 0 for file_name in filtered_labels]].sum().sum()
    total_plagiat_words = matrix[[file_name == 1 for file_name in filtered_labels]].sum()
    total_non_plagiat_words = matrix[[file_name == 0 for file_name in filtered_labels]].sum()

    word_totals = {}
    for word, index in featurizer.vocabulary.items():
        total = totals[index, 0]
        total_plagiat = totals[index, 1]
        #the likelihoods of each word == P(word/class) 
        word_totals[word] = {
            1: (total_plagiat + 1) / total_plagiat_words if total > 0 else 0,
            0: (total - total_plagiat + 1) / total_non_plagiat_words
        }

    return word_totals

def compute_word_probabilities_tfidf(filtered_labels: Dict[str, str], featurizer, matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    totals = np.zeros((len(featurizer.vocabulary), 2))  # TF-IDF sums for each class

    for file_name, row in zip(filtered_labels.keys(), matrix):
        for word, index in featurizer.vocabulary.items():
            if filtered_labels[file_name] == 0:
                totals[index, 1] += row[index]
            else:
                totals[index, 0] += row[index]

    total_plagiat_tfidf = totals[:, 1].sum()
    total_non_plagiat_tfidf = totals[:, 0].sum()

    word_totals = {}
    for word, index in featurizer.vocabulary.items():
        word_totals[word] = {
            1: (totals[index, 1] + 1) / (total_plagiat_tfidf + 1),  # Smooth with Laplace
            0: (totals[index, 0] + 1) / (total_non_plagiat_tfidf + 1),
        }

    return word_totals



if __name__ == "__main__":

    c = NaiveBayesClassifier()
    X = np.random.randint(1,10,(50,10))
    Xt = np.random.randint(1,10,(3,10))
    y = np.random.randint(0,2,(7,))

    c.fit(X,y)
    print(c.classes)
    print(c.class_priors)
    print(c.features_prob)
    print(c.predict(Xt))