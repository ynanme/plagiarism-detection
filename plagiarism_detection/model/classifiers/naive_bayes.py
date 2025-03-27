import numpy as np

from model.classifiers.classifier import Classifier

class NaiveBayesClassifier (Classifier):
    """
    Le classificateur Naive Bayes implémenté ici repose sur l'hypothèse d'indépendance des caractéristiques (les mots dans ce cas).
    Il utilise les probabilités a priori des classes et les probabilités conditionnelles des mots donnés une classe.
    """
    def __init__(self):
        """
        Initialisation du classificateur Naive Bayes.
        """
        super().__init__()
        self.class_priors = None
        self.word_probabilities = None
        self.vocab_size = None

    def fit(self, matrix, filtered_labels):
        """
        Apprend les paramètres du modèle (probabilités a priori et probabilités des mots pour chaque classe).

        Args:
            matrix (ndarray): Matrice de termes-documents où chaque ligne est un document et chaque colonne un mot du vocabulaire.
            filtered_labels (array): Étiquettes (classes) des documents.
        """
        super().fit(matrix, filtered_labels)
        self.vocab_size = matrix.shape[1]
        self.class_priors = self.compute_class_priors(matrix, filtered_labels)
        self.word_probabilities = self.compute_word_probabilities(matrix, filtered_labels)

    def predict(self, matrix):
        """
        Prédit les classes des documents à partir de la matrice de termes-documents.

        Args:
            matrix (ndarray ou list): Matrice de termes-documents à prédire (peut être une liste convertie en ndarray).

        Returns:
            ndarray: Tableau des indices des classes prédites pour chaque document.
        """
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
        """
        Calcule les probabilités a priori des classes à partir des labels.

        Args:
            matrix (ndarray): Matrice de termes-documents.
            filtered_labels (array): Étiquettes (classes) des documents.

        Returns:
            ndarray: Probabilités a priori des classes.
        """
        total_files = matrix.shape[0]
        num_classes = len(np.unique(filtered_labels))

        class_priors = np.zeros(num_classes)
        for cls in range(num_classes):
            class_priors[cls] = np.sum(filtered_labels == cls) / total_files

        return class_priors
    
    @staticmethod
    def compute_word_probabilities(matrix, filtered_labels):
        """
        Calcule les probabilités conditionnelles des mots donnés une classe.

        Args:
            matrix (ndarray): Matrice de termes-documents.
            filtered_labels (array): Étiquettes (classes) des documents.

        Returns:
            ndarray: Probabilités des mots pour chaque classe.
        """
        vocab_size = matrix.shape[1]
        word_probabilities = np.zeros((len(np.unique(filtered_labels)), vocab_size))

        for cls in range(len(np.unique(filtered_labels))):
            word_cls = matrix[filtered_labels == cls] 
            word_counts = np.sum(word_cls, axis=0)
            total_count = np.sum(word_counts) 
            word_probabilities[cls] = (word_counts + 1) / (total_count + vocab_size)

        return word_probabilities
