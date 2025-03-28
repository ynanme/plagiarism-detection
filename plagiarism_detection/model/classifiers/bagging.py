import numpy as np
import copy
from sklearn.utils import resample

from model.classifiers.classifier import Classifier


class BaggingClassifier (Classifier):

    def __init__(self, base_estimator, n_estimators=10):
        """
        Implémentation d'un Bagging Classifier.
        """
        """
        - base_estimator : Modèle de base (ex: arbre, SVM, régression logistique...)
        - n_estimators : Nombre de modèles dans l’ensemble
        """
        super().__init__()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        """ Entraîne plusieurs modèles sur des sous-échantillons bootstrap des données. """
        super().fit(X, y)
        self.models = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = resample(X, y, replace=True) # 
            model = copy.deepcopy(self.base_estimator)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        """ Agrège les prédictions des modèles par vote majoritaire. """
        predictions = np.array([model.predict(X) for model in self.models])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

