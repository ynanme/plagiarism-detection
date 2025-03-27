import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from collections import defaultdict

from util.util import partition, draw_plot, file_name


class ExperimentManager ():

    def __init__ (self, detectors={}, data=[[],[]]):
        self.detectors = detectors
        self.texts, self.categories = data
        self.split_data()

    def add_detector (self, detector, name):
        self.detectors[name] = detector

    def remove_detector (self, name):
        if name in self.detectors:
            del self.detectors[name]

    def set_data (self, data):
        self.texts, self.categories = data

    def split_data (self, training_data_weight=0.8):
        self.training_indexes, self.testing_indexes = partition(self.get_indexes(), [training_data_weight])

    def train_detectors (self, texts, categories):
        for detector in self.detectors.values():
            detector.train(texts, categories)
    
    def evaluate_detectors (self):
        texts, categories = self.get_texts(self.testing_indexes), self.get_categories(self.testing_indexes)
        results = {}
        for name, detector in self.detectors.items():
            predicted_categories = [detector.classify(text) for text in texts]
            results[name] = Metrics.scores(np.array(categories), np.array(predicted_categories))
        return results

    def get_indexes (self, count=None):
        count = len(self.texts) if count is None else count
        return [index for index in range(count)]

    def get_texts (self, indexes):
        return [self.texts[index] for index in indexes]

    def get_categories (self, indexes):
        return [self.categories[index] for index in indexes]



    def rank_detectors (self, training_data_weight=0.8, repeat=1):
        all_scores = []
        for _ in range(repeat):
            self.split_data(training_data_weight)
            self.train_detectors(self.get_texts(self.training_indexes), self.get_categories(self.training_indexes))
            scores = self.evaluate_detectors()
            scores_frame = pd.DataFrame(scores, index=["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"])
            all_scores.append(scores_frame)
        final_scores = pd.concat(all_scores).groupby(level=0).mean()
        final_rankings = final_scores.rank(axis=1, ascending=False, method="min").astype(int)
        print(final_scores)
        return final_scores, final_rankings


    def evaluate_detectors_on_learning (self, training_data_weight=0.8, repeat=1, batch_size_increment=10):
    
        # Initialisation du dictionnaire pour stocker les résultats de chaque répétition
        all_results = {name: [] for name in self.detectors.keys()}
        
        # Exécution de la logique pour chaque répétition
        for _ in range(repeat):
            results = {name: [] for name in self.detectors.keys()}  # Réinitialisation pour chaque répétition
            self.split_data(training_data_weight)  # Séparer les données
            
            for batch_size in range(batch_size_increment, len(self.training_indexes)+batch_size_increment, batch_size_increment):
                trained_on_texts = self.get_texts(self.training_indexes[:batch_size])
                trained_on_texts_categories = self.get_categories(self.training_indexes[:batch_size])
                self.train_detectors(trained_on_texts, trained_on_texts_categories)
                
                scores = self.evaluate_detectors()  # Évaluer les détecteurs
                for name in scores:
                    results[name].append(scores[name])  # Ajouter les scores à la liste
            
            # Ajouter les résultats de cette répétition à all_results
            for name, scores in results.items():
                all_results[name].append(scores)
        
        # Calcul des moyennes des scores sur toutes les répétitions
        avg_results = {}
        for name, scores in all_results.items():
            avg_results[name] = np.mean(scores, axis=0)  # Moyenne des scores pour chaque détecteur

        return avg_results,all_results
                


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
        #correct_predictions = np.sum(y_true == y_pred)
        #total_predictions = len(y_true)
        #return correct_predictions / total_predictions if total_predictions > 0 else 0.0
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
    def scores (y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.array([
            Metrics.accuracy(y_true, y_pred),
            Metrics.balanced_accuracy(y_true, y_pred),
            Metrics.precision(y_true, y_pred),
            Metrics.recall(y_true, y_pred),
            Metrics.f1_score(y_true, y_pred)
        ])

    
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

    @staticmethod
    def plot_scores_and_rankings (scores, rankings):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        scores.T.plot(kind="bar", ax=axes[0])
        axes[0].set_title("Scores moyens des détecteurs")
        axes[0].set_ylabel("Score")
        axes[0].set_xlabel("Détecteurs")
        axes[0].legend(title="Métriques")
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)

        rankings.T.plot(kind="bar", ax=axes[1])
        axes[1].invert_yaxis()
        axes[1].set_title("Classement des détecteurs")
        axes[1].set_ylabel("Rang (1 = meilleur)")
        axes[1].set_xlabel("Détecteurs")
        axes[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[1].legend(title="Métriques")
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_learning_curves(results, batch_size_increment):
        metrics_names = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score"]
        
        # Nombre de batchs
        num_batches = len(next(iter(results.values())))  # Prend la taille d'un détecteur
        
        # Axe des X (taille des données utilisées pour l'entraînement)
        x_values = np.arange(batch_size_increment, (num_batches + 1) * batch_size_increment, batch_size_increment)

        # Crée une figure avec 5 sous-graphiques (1 ligne, 5 colonnes)
        fig, axes = plt.subplots(1, 5, figsize=(18, 6))
        plt.tight_layout(pad=5.0)
    
        
        for metric_idx, (metric_name, ax) in enumerate(zip(metrics_names, axes)):
            for detector_name, scores in results.items():
                metric_values = [score[metric_idx] for score in scores]  # Extraire la métrique correspondante
                ax.plot(x_values, metric_values, label=detector_name)
            
            ax.set_title(metric_name)
            ax.set_xlabel("Taille des données d'entraînement")
            ax.set_ylabel("Score")
            ax.grid(True)
            ax.legend(loc='upper left', fontsize=8)
        
        plt.show()