import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from util.util import partition, draw_plot, file_name


class ExperimentManager ():

    def __init__ (self, detectors={}, data=[[],[]], saving_results_path=None):
        self.detectors = {}
        self.texts, self.categories = data
        self.saving_results_path = saving_results_path
        if saving_results_path is not None and not self.saving_results_path.endswith("/"):
            self.saving_results_path += "/"

    def add_detector (self, detector, name):
        self.detectors[name] = detector

    def remove_detector (self, name):
        if name in self.detectors:
            del self.detectors[name]

    def set_data (self, data):
        self.texts, self.categories = data

    def train_detectors (self, texts, categories):
        #categories = {index: category for index,category in enumerate(categories)}
        for detector in self.detectors.values():
            detector.train(texts, categories)
    
    def evaluate_detectors (self, texts, categories):
        #categories = {index: category for index,category in enumerate(categories)}
        accuracies = []
        for detector in self.detectors.values():
            accuracies.append(detector.evaluate(texts, categories))
        return accuracies

    def get_indexes (self, count=None):
        count = len(self.texts) if count is None else count
        return [index for index in range(count)]

    def filter_indexes (self, indexes, filterr):
        if filterr is not None:
            return [index for index in indexes if self.categories[index] == filterr]
        return indexes

    def get_filters_description (self, filter_on_training, filter_on_testing):
        filters_description = ""
        filters_description += f"trained on {filter_on_training} " if filter_on_training is not None else ""
        filters_description += f"tested on {filter_on_testing}" if filter_on_testing is not None else ""
        return filters_description
    
    def get_names_extension (self):
        return "&".join(self.detectors.keys())

    
    def run_experiment (self, training_indexes, testing_indexes=None, repeat=1, description="Experiment"):
        run_on_trained = testing_indexes is None
        testing_texts = [] if run_on_trained else [self.texts[index] for index in testing_indexes]
        testing_texts_categories = [] if run_on_trained else [self.categories[index] for index in testing_indexes]
        all_accuracies = []
        for _ in range(repeat):
            accuracies = []
            trained_on_texts, trained_on_texts_categories = [], []
            
            for index in training_indexes:
                trained_on_texts.append(self.texts[index])
                trained_on_texts_categories.append(self.categories[index])
                self.train_detectors(trained_on_texts, trained_on_texts_categories)
                accuracies.append(self.evaluate_detectors(
                    trained_on_texts if run_on_trained else testing_texts,
                    trained_on_texts_categories if run_on_trained else testing_texts_categories
                ))
            all_accuracies.append(accuracies)
        accuracies = np.mean(np.array(all_accuracies), axis=0)
        """draw_plot(
            [x for x in range(len(training_indexes))],
            [[accuracy[index] for accuracy in accuracies] for index in range(len(self.detectors))],
            self.detectors.keys(), ["Amount of data trained on", "Detector accuracy"], f"{description}",
            True, f"{self.saving_results_path}{file_name(description)}_{self.get_names_extension()}.png" if self.saving_results_path is not None else None
        )"""
        return self.get_indexes(len(training_indexes)), accuracies

    def run_on_all (self, training_data_weight=1, filter_on_training=None, filter_on_testing=None, repeat=1):
        all_indexes = self.get_indexes()
        training_indexes = partition(all_indexes, [training_data_weight])[0]
        return self.run_experiment(
            self.filter_indexes(training_indexes, filter_on_training),
            self.filter_indexes(all_indexes, filter_on_testing),
            repeat,
            f"Evaluating on all data, training data is {round(training_data_weight*100)}%, {self.get_filters_description(filter_on_training, filter_on_testing)}, average on {repeat} repeats"
        )

    def run_on_trained (self, training_data_weight=1, filter_on_training=None, filter_on_testing=None, repeat=1):
        training_indexes = partition(self.get_indexes(), [training_data_weight])[0]
        #return self.run_experiment(training_indexes, description=f"Evaluating on trained data, training data is {round(training_data_weight*100)}%")
        return self.run_experiment(
            self.filter_indexes(training_indexes, filter_on_training),
            repeat=repeat,
            description=f"Evaluating on trained data, training data is {round(training_data_weight*100)}%, {self.get_filters_description(filter_on_training, filter_on_testing)}, average on {repeat} repeats"
        )

    def run_on_untrained (self, training_data_weight=0.8, filter_on_training=None, filter_on_testing=None, repeat=1):
        training_indexes, testing_indexes = partition(self.get_indexes(), [training_data_weight])
        #return self.run_experiment(training_indexes, testing_indexes, f"Evaluating on untrained data, training data is {round(training_data_weight*100)}%")
        return self.run_experiment(
            self.filter_indexes(training_indexes, filter_on_training),
            self.filter_indexes(testing_indexes, filter_on_testing),
            repeat,
            f"Evaluating on untrained data, training data is {round(training_data_weight*100)}%, {self.get_filters_description(filter_on_training, filter_on_testing)}, average on {repeat} repeats"
        )


    def __str__ (self):
        return f"EXPERIMENTS MANAGER ON {str(detector for detector in self.detectors.values())}"




class Metrics:
    """
    Classe contenant des méthodes pour évaluer les performances d'un modèle de classification.
    """

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice de confusion pour un problème de classification binaire.

        Args:
            y_true (np.ndarray): Classes réelles.
            y_pred (np.ndarray): Classes prédites.

        Returns:
            np.ndarray: Matrice de confusion sous la forme [[TN, FP], [FN, TP]].
        """
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TP = np.sum((y_true == 1) & (y_pred == 1))
        return np.array([[TN, FP], [FN, TP]])

    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule l'accuracy (taux de bonnes prédictions).

        Args:
            y_true (np.ndarray): Classes réelles.
            y_pred (np.ndarray): Classes prédites.

        Returns:
            float: Score d'accuracy.
        """
        cm = Metrics.confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()
        return (TP + TN) / (TP + TN + FP + FN)

    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule la précision (taux de vrais positifs parmi les prédictions positives).

        Args:
            y_true (np.ndarray): Classes réelles.
            y_pred (np.ndarray): Classes prédites.

        Returns:
            float: Score de précision.
        """
        cm = Metrics.confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule le rappel (taux de vrais positifs détectés).

        Args:
            y_true (np.ndarray): Classes réelles.
            y_pred (np.ndarray): Classes prédites.

        Returns:
            float: Score de rappel.
        """
        cm = Metrics.confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule le F1-score (moyenne harmonique entre précision et rappel).

        Args:
            y_true (np.ndarray): Classes réelles.
            y_pred (np.ndarray): Classes prédites.

        Returns:
            float: Score F1.
        """
        precision = Metrics.precision(y_true, y_pred)
        recall = Metrics.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    
    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule la balanced accuracy (accuracy équilibrée).

        Args:
            y_true (np.ndarray): Classes réelles.
            y_pred (np.ndarray): Classes prédites.

        Returns:
            float: Score de balanced accuracy.
        """
        cm = Metrics.confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()
        recall_pos = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        return (recall_pos + recall_neg) / 2

    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Affiche la matrice de confusion sous forme de heatmap.

        Args:
            y_true (np.ndarray): Vecteur des vraies classes.
            y_pred (np.ndarray): Vecteur des classes prédites.
        """
        cm = Metrics.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non Plagié", "Plagié"], yticklabels=["Non Plagié", "Plagié"])
        plt.xlabel("Prédiction du modèle")
        plt.ylabel("Classe réelle")
        plt.title("Matrice de Confusion")
        plt.show()
