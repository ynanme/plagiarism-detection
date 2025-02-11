import numpy as np

from model.classifiers.classifier import Classifier


class PerceptronClassifier (Classifier):
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, matrix, binary_labels):
        nb_docs, nb_vocab = matrix.shape
        self.weights = np.zeros(nb_vocab)  # Initialize weights to zero

        for epoch in range(self.epochs):
            #print(f"Epoch {epoch} weight {self.weights} bias {self.bias}")
            for i in range(nb_docs):
                z = np.dot(matrix[i], self.weights) + self.bias
                if - binary_labels[i] * z > 0:
                    # Update weights and bias
                    self.weights += self.learning_rate * binary_labels[i] * matrix[i]
                    self.bias += self.learning_rate * binary_labels[i]
                    

    def predict(self, matrix):
        z = np.dot(matrix, self.weights) + self.bias
        #print("linear_output", np.sign(linear_output))
        predictions = np.sign(z)
        return predictions