import numpy as np
import copy

from model.classifiers.classifier import Classifier



class AdaBoost (Classifier):

    def __init__ (self, base_model, models_count=50, labels_mapping_function=None):
        super().__init__()
        self.base_model = base_model
        self.models = []
        self.models_weights = []
        self.models_count = models_count
        self.labels_mapping_function = labels_mapping_function

    
    def fit (self, features_matrix, labels):

        super().fit(features_matrix, labels)

        self.models = []
        self.models_weights = []

        samples_count, _ = features_matrix.shape
        samples_weights = np.ones(samples_count) / samples_count
        normalized_labels = self.map_labels(labels)

        for _ in range(self.models_count):
            current_model = copy.deepcopy(self.base_model)
            current_model.fit(features_matrix, normalized_labels, sample_weight=samples_weights)
            predicted_labels = current_model.predict(features_matrix)
            self.validate_labels(normalized_labels, [-1, 1])
            self.validate_labels(predicted_labels, [-1, 1])
            current_model_error = np.sum(samples_weights * (predicted_labels != normalized_labels))
            if current_model_error <= 0:
                current_model_weight = 1
                self.models.append(current_model)
                self.models_weights.append(current_model_weight)
                break
            current_model_weight = 1/2 * np.log((1-current_model_error)/current_model_error)
            samples_weights *= np.exp(-(predicted_labels * normalized_labels) * current_model_weight)
            samples_weights /= np.sum(samples_weights)
            self.models.append(current_model)
            self.models_weights.append(current_model_weight)

    
    def predict (self, features_matrix):
        samples_count, _ = features_matrix.shape
        predictions = np.zeros(samples_count)
        for i in range(samples_count):
            prediction = 0
            for model, model_weight in zip(self.models, self.models_weights):
                prediction += (model.predict(features_matrix)[i] * model_weight)
            predictions[i] = prediction
        predicted_labels = self.map_labels(np.sign(predictions), reverse=True)
        self.validate_labels(predicted_labels, [0, 1])
        return predicted_labels
    

    def map_labels (self, labels, reverse=False):
        if self.labels_mapping_function is None:
            return labels
        return np.array([self.labels_mapping_function(label, reverse) for label in labels])

    def validate_labels (self, labels, expected_labels):
        for label in labels:
            if label not in expected_labels:
                raise ValueError(f"unexpected label {label} found")

