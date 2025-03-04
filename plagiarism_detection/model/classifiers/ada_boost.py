import numpy as np

from model.classifiers.classifier import Classifier



class AdaBoost (Classifier):

    def __init__ (self, base_model, classes=np.array([]), models_count=500, max_error_tolerated=0.5, labels_mapping_function=None):
        super().__init__(classes)
        self.base_model = base_model
        self.models = []
        self.models_weights = []
        self.models_count = models_count
        self.max_error_tolerated = max_error_tolerated
        self.labels_mapping_function = labels_mapping_function

    
    def fit (self, features_matrix, labels):

        samples_count, _ = features_matrix.shape
        samples_weights = np.ones(samples_count) / samples_count

        for _ in range(self.models_count):
            current_model = self.base_model
            current_model.fit(features_matrix, self.map_labels(labels), sample_weight=samples_weights)
            predicted_labels = current_model.predict(features_matrix)
            current_model_error = np.sum(samples_weights[predicted_labels != labels])
            if current_model_error <= self.max_error_tolerated:
                current_model_weight = 1/2 * np.log((1-current_model_error)/current_model_error)
                samples_weights *= np.exp(predicted_labels * labels * current_model_weight)
                #samples_weights[predicted_labels == labels] *= np.exp(-current_model_weight)
                #samples_weights[predicted_labels != labels] *= np.exp(current_model_weight)
                self.models.append(current_model)
                self.models_weights.append(current_model_weight)

    
    def predict (self, features_matrix):
        samples_count, _ = features_matrix.shape
        predictions = np.zeros(samples_count)
        for i in range(samples_count):
            prediction = 0
            for model, model_weight in zip(self.models, self.models_weights):
                prediction += (model.predict(features_matrix) * model_weight)
            predictions[i] = prediction
        return self.map_labels(np.sign(predictions))
    

    def map_labels (self, labels):
        if self.labels_mapping_function is None:
            return labels
        return np.array([self.labels_mapping_function(label) for label in labels])