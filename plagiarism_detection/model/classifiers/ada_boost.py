import numpy as np
from sklearn.base import clone
import copy
from model.classifiers.classifier import Classifier



class AdaBoost (Classifier):

    def __init__ (self, base_model, classes=np.array([]), models_count=25, labels_mapping_function=None):
        super().__init__(classes)
        self.base_model = base_model
        self.models = []
        self.models_weights = []
        self.models_count = models_count
        self.labels_mapping_function = labels_mapping_function

    
    def fit (self, features_matrix, labels):
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
                prediction += (model.predict(features_matrix) * model_weight)
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



from sklearn.utils import resample

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators=10):
        """
        Implémentation d'un Bagging Classifier.
        
        - base_estimator : Modèle de base (ex: arbre, SVM, régression logistique...)
        - n_estimators : Nombre de modèles dans l’ensemble
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        """ Entraîne plusieurs modèles sur des sous-échantillons bootstrap des données. """
        self.models = []  # Liste pour stocker les modèles

        for _ in range(self.n_estimators):
            # 📌 Créer un sous-échantillon bootstrap
            X_sample, y_sample = resample(X, y, replace=True)
            
            # 📌 Cloner et entraîner un modèle sur cet échantillon
            model = self.base_estimator()  # Créer une nouvelle instance du modèle
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        """ Agrège les prédictions des modèles par vote majoritaire. """
        predictions = np.array([model.predict(X) for model in self.models])
        # 📌 Vote majoritaire : On prend la classe la plus prédite par les modèles
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)