import numpy as np

class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, epochs=100, threshold=0.5, patience=5, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.threshold = threshold

        self.tolerance = tolerance  
        self.patience = patience 

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, matrix, binary_labels):
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
        z = np.dot(matrix, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        #print(probabilities)
        return (probabilities >= self.threshold).astype(int)

