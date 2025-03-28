import numpy as np
import pandas as pd

from experimentations.metrics import Metrics
from util.util import partition


class ExperimentManager ():

    def __init__ (self, detectors={}, data=[[],[]]):
        self.detectors = detectors
        self.texts, self.categories = data
        self.training_indexes, self.testing_indexes = [], []

    def add_detector (self, detector, name):
        self.detectors[name] = detector

    def remove_detector (self, name):
        if name in self.detectors:
            del self.detectors[name]

    def remove_all_detectors (self):
        self.detectors = {}

    def set_data (self, data):
        self.texts, self.categories = data

    def split_data (self, training_data_weight=0.8):
        self.training_indexes, self.testing_indexes = partition(self.get_indexes(), [training_data_weight])

    def train_detectors (self, texts, categories):
        for detector in self.detectors.values():
            detector.train(texts, categories)
    
    def evaluate_detectors (self):
        texts, categories = self.get_texts(self.testing_indexes), self.get_categories(self.testing_indexes)
        results = {}
        for name, detector in self.detectors.items():
            predicted_categories = [detector.classify(text) for text in texts]
            results[name] = Metrics.scores(np.array(categories), np.array(predicted_categories))
        return results

    def get_indexes (self, count=None):
        count = len(self.texts) if count is None else count
        return [index for index in range(count)]

    def get_texts (self, indexes):
        return [self.texts[index] for index in indexes]

    def get_categories (self, indexes):
        return [self.categories[index] for index in indexes]



    def rank_detectors (self, training_data_weight=0.8, repeat=1):
        all_scores = []
        for _ in range(repeat):
            self.split_data(training_data_weight)
            self.train_detectors(self.get_texts(self.training_indexes), self.get_categories(self.training_indexes))
            scores = self.evaluate_detectors()
            scores_frame = pd.DataFrame(scores, index=["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"])
            all_scores.append(scores_frame)
        final_scores = pd.concat(all_scores).groupby(level=0).mean()
        final_rankings = final_scores.rank(axis=1, ascending=False, method="min").astype(int)
        return final_scores, final_rankings


    def compute_accuracies_for_confusion_matrixs (self, training_data_weight=0.8):
        self.split_data(training_data_weight)
        self.train_detectors(self.get_texts(self.training_indexes), self.get_categories(self.training_indexes))
        texts = self.get_texts(self.testing_indexes)
        y_true = self.get_categories(self.testing_indexes)
        cms = {}
        for name in self.detectors:
            y_pred = [self.detectors[name].classify(text) for text in texts]
            cms[name] = (np.array(y_true), np.array(y_pred))
        return cms


    def evaluate_detectors_on_learning (self, training_data_weight=0.8, batch_size_increment=10):
    
        results = {name: [] for name in self.detectors.keys()}
        self.split_data(training_data_weight)
        
        for batch_size in range(batch_size_increment, len(self.training_indexes)+batch_size_increment, batch_size_increment):
            trained_on_texts = self.get_texts(self.training_indexes[:batch_size])
            trained_on_texts_categories = self.get_categories(self.training_indexes[:batch_size])
            self.train_detectors(trained_on_texts, trained_on_texts_categories)
            
            scores = self.evaluate_detectors()
            for name in scores:
                results[name].append(scores[name])

        return results
                


    def __str__ (self):
        return f"EXPERIMENTS MANAGER ON {str(detector for detector in self.detectors.values())}"


