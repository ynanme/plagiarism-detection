import numpy as np

from model.classifiers.classifier import Classifier

class LogisticRegressionClassifier (Classifier):
    """
    Ce classificateur implémente une régression logistique pour la classification binaire.
    La régression logistique modélise la probabilité d'une classe en fonction des caractéristiques d'entrée.
    """
    def __init__(self, learning_rate=0.01, epochs=100, threshold=0.5, patience=5, tolerance=1e-4):
        """
        Initialisation du classificateur de régression logistique.

        Args:
            learning_rate (float): Taux d'apprentissage pour la mise à jour des poids (par défaut 0.01).
            epochs (int): Nombre d'itérations d'entraînement (par défaut 100).
            threshold (float): Seuil pour la classification (par défaut 0.5).
            patience (int): Nombre d'époques sans amélioration avant d'arrêter l'entraînement (par défaut 5).
            tolerance (float): Critère de tolérance pour la convergence du modèle (par défaut 1e-4).
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.threshold = threshold

        self.tolerance = tolerance  
        self.patience = patience 

    def sigmoid(self, z):
        """
        Fonction sigmoïde utilisée pour obtenir des probabilités de classe.

        Args:
            z (ndarray): Le produit scalaire entre les caractéristiques et les poids.

        Returns:
            ndarray: Probabilités calculées à partir de la fonction sigmoïde.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, matrix, binary_labels):
        """
        Entraîne le modèle sur les données d'entrée (matrice de termes-documents) et les étiquettes binaires.

        Args:
            matrix (ndarray): Matrice de caractéristiques, où chaque ligne représente un document et chaque colonne un mot.
            binary_labels (ndarray): Étiquettes binaires pour chaque document (0 ou 1).
        """
        super().fit(matrix, binary_labels)
        nb_docs, nb_vocab = matrix.shape
        self.weights = np.zeros(nb_vocab)

        best_loss = float("inf")
        no_improvement_count = 0

        for epoch in range(self.epochs):
            z = np.dot(matrix, self.weights) + self.bias
            predictions = self.sigmoid(z)
            loss = -np.mean(binary_labels * np.log(predictions + 1e-9) + (1 - binary_labels) * np.log(1 - predictions + 1e-9))

            dw = np.dot(matrix.T, (predictions - binary_labels))
            db = np.sum(predictions - binary_labels) 

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if best_loss - loss < self.tolerance:
                no_improve_count += 1
            else:
                best_loss = loss
                no_improve_count = 0

            if no_improve_count >= self.patience:
                break

    def predict(self, matrix):
        """
        Prédit les classes pour de nouveaux documents en fonction des caractéristiques d'entrée.

        Args:
            matrix (ndarray): Matrice de caractéristiques des nouveaux documents.

        Returns:
            ndarray: Tableau des classes prédites (0 ou 1) pour chaque document.
        """
        z = np.dot(matrix, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        #print(probabilities)
        return (probabilities >= self.threshold).astype(int)


    def __str__ (self):
        return f"LogisticRegression(learning_rate={self.learning_rate}, epochs={self.epochs}, threshold={self.threshold}, patience={self.patience}, tolerance={self.tolerance})"