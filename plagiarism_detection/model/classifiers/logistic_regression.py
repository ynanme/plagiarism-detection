import numpy as np

from model.classifiers.classifier import Classifier


class LogisticRegressionClassifier (Classifier):
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, matrix, binary_labels):
        nb_docs, nb_vocab = matrix.shape
        self.weights = np.zeros(nb_vocab)
        self.bias = 0

        for epoch in range(self.epochs):
            z = np.dot(matrix, self.weights) + self.bias
            predictions = self.sigmoid(z)
            #loss = -np.mean(binary_labels * np.log(predictions) + (1 - binary_labels) * np.log(1 - predictions))
            #print(f"Epoch {epoch}: Loss = {loss}")  
            dw = np.dot(matrix.T, (predictions - binary_labels))
            db = predictions - binary_labels

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, matrix):
        z = np.dot(matrix, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        return (probabilities >= 0.5).astype(int)