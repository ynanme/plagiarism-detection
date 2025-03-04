import numpy as np

class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.00001, epochs=300):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, matrix, binary_labels):
        nb_docs, nb_vocab = matrix.shape
        self.weights = np.zeros(nb_vocab)

        for epoch in range(self.epochs):
            z = np.dot(matrix, self.weights) + self.bias
            predictions = self.sigmoid(z)

            dw = np.dot(matrix.T, (predictions - binary_labels))
            db = np.sum(predictions - binary_labels) 

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, matrix):
        z = np.dot(matrix, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        print(probabilities)
        return (probabilities >= 0.71).astype(int)
