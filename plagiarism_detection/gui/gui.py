import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 📌 1. Générer un dataset bidimensionnel (X1, X2)
X, y = make_classification(
    n_samples=300, 
    n_features=5,      # Total des features
    n_informative=3,   # Features utiles pour la classification
    n_redundant=1,     # Features combinées linéairement
    n_repeated=0,      # Features répétées
    random_state=42
)
# 📌 2. Séparer en train/test
# 📌 1. Générer un dataset avec 2 FEATURES (pour qu'on puisse le visualiser en 2D)
X, y = make_classification(n_samples=300, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=42)

# 📌 2. Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 3. Entraîner un Random Forest avec exactement 2 FEATURES
clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
clf.fit(X_train, y_train)  # Entraînement avec 2 features
# 📌 4. Créer une grille de points pour tracer la frontière de décision
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# 📌 5. Prédire la classe pour chaque point de la grille
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 📌 6. Tracer la frontière de décision
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)  # Colore les régions
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolors='k')  # Points d'entraînement
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Frontière de décision du Random Forest")
plt.show()
