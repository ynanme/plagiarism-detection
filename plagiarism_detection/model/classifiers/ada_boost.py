import numpy as np

from model.classifiers.classifier import Classifier



class AdaBoost (Classifier):

    def __init__ (self, base_model, classes=np.array([]), models_count=50, max_error_tolerated=0.5, labels_mapping_function=None):
        super().__init__(classes)
        self.base_model = base_model
        self.models = np.array([])
        self.models_weights = np.array([])
        self.max_error_tolerated = max_error_tolerated
        self.labels_mapping_function = labels_mapping_function

    
    def fit (features_matrix, labels):

        samples_count, _ = features_matrix.shape
        samples_weights = np.ones(samples_count) / samples_count

        for _ in range(self.models_count):
            current_model = self.base_model
            current_model.fit(features_matrix, labels, samples_weights=samples_weights)
            predicted_labels = current_model.predict(features_matrix)
            current_model_error = np.sum(samples_weights[predicted_labels != labels])
            if current_model_error <= self.max_error_tolerated:
                current_model_weight = 1/2 * np.log((1-current_model_error)/current_model_error)
                samples_weights[predicted_labels == labels] *= np.exp(-current_model_weight)
                samples_weights[predicted_labels != labels] *= np.exp(current_model_weight)
                self.models.append(current_model)
                self.models_weights.append(current_model_weight)

    
    def predict (features_matrix):
        samples_count, _ = features_matrix.shape
        predictions = np.zeros(samples_count)
        for i in range(samples_count):
            prediction = 0
            for model, model_weight in zip(self.models, self.models_weights):
                prediction += (model.predict(features_matrix) * model_weight)
            predictions.append(predictions)
        if self.labels_mapping_function is not None:
            return self.labels_mapping_function(np.sign(predictions))
        return np.sign(predictions)