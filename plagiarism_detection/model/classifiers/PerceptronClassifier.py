import numpy as np

class PerceptronClassifier:
    def __init__(self, learning_rate=0.00000000001, epochs=100, tolerance=1e-4, patience=5):
    #def __init__(self, learning_rate=0.01, epochs=100, tolerance=1e-4, patience=5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

        self.tolerance = tolerance  
        self.patience = patience 

    def fit(self, matrix, binary_labels):
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
                print(f"Early stopping at epoch {epoch+1}")
                break 

            best_loss = avg_loss  

    def predict(self, matrix):
        z = np.dot(matrix, self.weights) + self.bias
        predictions = np.sign(z)
        #print(predictions)
        return predictions
