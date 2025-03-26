from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 📌 Prédictions du modèle (exemple)
y_test = [1, 0, 1, 1, 0, 0, 1, 1, 0, 0]  # 1 = Plagié, 0 = Original
y_pred = [1, 0, 1, 0, 0, 0, 1, 1, 1, 0]  # Prédictions du modèle

# 📌 Calcul des métriques
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 📌 Affichage des résultats
print(f"Accuracy : {accuracy:.2f}")
print(f"Précision : {precision:.2f}")
print(f"Rappel : {recall:.2f}")
print(f"F1-score : {f1:.2f}")

# 📌 Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Original", "Plagié"], yticklabels=["Original", "Plagié"])
plt.xlabel("Prédiction du modèle")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion - Détection de Plagiat")
plt.show()
