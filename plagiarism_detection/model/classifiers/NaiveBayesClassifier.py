import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = None
        self.word_probabilities = None
        self.vocab_size = None

    def fit(self, matrix, filtered_labels):
        self.vocab_size = matrix.shape[1]
        self.class_priors = self.compute_class_priors(matrix, filtered_labels)
        self.word_probabilities = self.compute_word_probabilities(matrix, filtered_labels)

    def predict(self, matrix):
        if isinstance(matrix, list):
            matrix = np.array([matrix])

        total_files = matrix.shape[0]
        nb_classes = len(self.class_priors)

        log_probs = np.zeros((total_files, nb_classes))

        for cls in range(nb_classes):
            log_probs[:, cls] = np.log(self.class_priors[cls])

            for i in range(total_files):
                log_probs[i, cls] += np.sum(matrix[i] * np.log(self.word_probabilities[cls]))

        return np.argmax(log_probs, axis=1)

    @staticmethod
    def compute_class_priors(matrix, filtered_labels):
        total_files = matrix.shape[0]
        num_classes = len(np.unique(filtered_labels))

        class_priors = np.zeros(num_classes)
        for cls in range(num_classes):
            class_priors[cls] = np.sum(filtered_labels == cls) / total_files

        return class_priors
    
    @staticmethod
    def compute_word_probabilities(matrix, filtered_labels):
        vocab_size = matrix.shape[1]
        word_probabilities = np.zeros((len(np.unique(filtered_labels)), vocab_size))

        for cls in range(len(np.unique(filtered_labels))):
            word_cls = matrix[filtered_labels == cls] 
            word_counts = np.sum(word_cls, axis=0)
            total_count = np.sum(word_counts) 
            word_probabilities[cls] = (word_counts + 1) / (total_count + vocab_size)

        return word_probabilities
