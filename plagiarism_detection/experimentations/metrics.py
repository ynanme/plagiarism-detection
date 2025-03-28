import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

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
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title="Matrice de Confusion", saving_results_path=None):
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
        plt.title(title)
        if saving_results_path is not None:
            plt.savefig(saving_results_path)
        plt.show()

    @staticmethod
    def plot_scores_and_rankings (scores, rankings, saving_results_path=None):
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
        if saving_results_path is not None:
            fig.savefig(saving_results_path)
        plt.show()


    @staticmethod
    def plot_learning_curves(results, batch_size_increment, saving_results_path=None):
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
        
        if saving_results_path is not None:
            fig.savefig(saving_results_path)
        plt.show()