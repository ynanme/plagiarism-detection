import numpy as np

from model.classifiers.classifier import Classifier

class PerceptronClassifier (Classifier):
    """
    Ce classificateur implémente un perceptron pour la classification binaire. 
    Le perceptron est un modèle linéaire d'apprentissage supervisé.
    """
    def __init__(self, learning_rate=0.00000000001, epochs=100, patience=5, tolerance=1e-4):
        """
        Initialisation du classificateur Perceptron.

        Args:
            learning_rate (float): Taux d'apprentissage pour la mise à jour des poids (par défaut 0.00000000001).
            epochs (int): Nombre d'itérations d'entraînement (par défaut 100).
            patience (int): Nombre d'époques sans amélioration avant d'arrêter l'entraînement (par défaut 5).
            tolerance (float): Critère de tolérance pour la convergence du modèle (par défaut 1e-4).
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

        self.tolerance = tolerance  
        self.patience = patience 

    def fit(self, matrix, binary_labels):
        """
        Entraîne le perceptron sur les données d'entrée (matrice de termes-documents) et les étiquettes binaires.

        Args:
            matrix (ndarray): Matrice de caractéristiques, où chaque ligne représente un document et chaque colonne un mot.
            binary_labels (ndarray): Étiquettes binaires pour chaque document (0 ou 1).
        """
        super().fit(matrix, binary_labels)
        nb_docs, nb_vocab = matrix.shape
        self.weights = np.random.rand(nb_vocab)

        best_loss = float("inf")
        no_improvement_count = 0

        for epoch in range(self.epochs):
            total_loss = 0 

            for i in range(nb_docs):
                z = np.dot(matrix[i], self.weights) + self.bias
                loss = max(0, -binary_labels[i] * z)
                total_loss += loss

                if -binary_labels[i] * z > 0:
                    self.weights += self.learning_rate * binary_labels[i] * matrix[i]
                    self.bias += self.learning_rate * binary_labels[i]

            avg_loss = total_loss / nb_docs
            #print(f"Epoch {epoch+1}, Loss: {avg_loss}")

            if abs(best_loss - avg_loss) < self.tolerance:
                no_improvement_count += 1
            else:
                no_improvement_count = 0  

            if no_improvement_count >= self.patience:
                #print(f"Early stopping at epoch {epoch+1}")
                break 

            best_loss = avg_loss  

    def predict(self, matrix):
        """
        Prédit les classes pour de nouveaux documents en fonction des caractéristiques d'entrée.

        Args:
            matrix (ndarray): Matrice de caractéristiques des nouveaux documents.

        Returns:
            ndarray: Tableau des classes prédites (-1 ou 1) pour chaque document.
        """
        z = np.dot(matrix, self.weights) + self.bias
        predictions = np.sign(z)
        #print(predictions)
        return predictions


    def __str__ (self):
        return f"Perceptron(learning_rate={self.learning_rate}, epochs={self.epochs}, patience={self.patience}, tolerance={self.tolerance})"