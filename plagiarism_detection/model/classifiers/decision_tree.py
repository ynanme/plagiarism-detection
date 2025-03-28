import numpy as np

from model.classifiers.classifier import Classifier

class Node:
    def __init__(self, data=None, children=None, split_on = None, pred_class=None, is_leaf=False):
        """
        Initialise un noeud dans un arbre de décision.

        Arguments:
        data (ndarray) : Les données contenues dans ce noeud (par défaut None).
        children (dict,) : Les enfants du noeud (par défaut None).
        split_on (int) : L'indice de la caractéristique sur laquelle le noeud divise (par défaut None).
        pred_class (int) : La classe prédite pour ce noeud si c'est une feuille (par défaut None).
        is_leaf (bool) : Indique si ce noeud est une feuille (par défaut False).
        """
        self.data = data
        self.children = children
        self.split_on = split_on
        self.pred_class = pred_class
        self.is_leaf = is_leaf

class DecisionTree (Classifier):
    def __init__(self, max_depth=5, criterion='entropy'):
        """
        Initialise le classifieur arbre de décision.

        Arguments:
        max_depth (int) : La profondeur maximale de l'arbre (par défaut 5).
        criterion (str) : Critère pour calculer l'impureté ('entropy' ou 'gini', par défaut 'entropy').
        """
        super().__init__()
        self.root = Node()
        self.max_depth = max_depth
        self.criterion = criterion
    
    def fit(self, X, Y):
        """
        Entraîne le classifieur sur les données d'entraînement.

        Arguments:
        X (ndarray) : Les caractéristiques des données d'entraînement.
        Y (ndarray) : Les étiquettes des données d'entraînement.
        """
        super().fit(Y, Y)
        data = np.column_stack([X, Y])
        self.root.data = data
        #print(np.unique(Y, return_counts=True))
        self.best_split(self.root)

    def predict(self, X):
        """
        Prédit les classes pour les données d'entrée.

        Arguments:
        X (ndarray) : Les données pour lesquelles prédire les classes.

        Retourne:
        ndarray : Les prédictions des classes pour chaque instance.
        """
        #print("in decision", len(X), len(X[0]))
        predictions = np.array([self.traverse_tree(x, self.root) for x in X])
        #self.print_tree()
        #print(predictions)
        return predictions

    def traverse_tree(self, x, node):
        """
        Traverse l'arbre de décision pour prédire la classe d'une instance donnée.

        Arguments:
        x (ndarray) : Une instance de données.
        node (Node) : Le noeud actuel.

        Retourne:
        int : La classe prédite pour l'instance.
        """
        if node.is_leaf:
            #print(node.pred_class)
            return node.pred_class

        feat_value = x[node.split_on]
        if feat_value not in node.children:  
            return node.pred_class
        
        predicted_class = self.traverse_tree(x, node.children[feat_value])

        return predicted_class

    def split_on_feature(self, data, feat_index):
        """
        Effectue la division sur une caractéristique donnée pour calculer l'impureté.

        Arguments:
        data (ndarray) : Les données à diviser.
        feat_index (int) : L'indice de la caractéristique sur laquelle diviser.

        Retourne:
        tuple : Un dictionnaire des enfants (split_nodes) et l'impureté pondérée (weighted_impurity).
        """
        feature_values = data[:, feat_index]
        unique_values = np.unique(feature_values)
        #print("unique", unique_values)
        split_nodes = {}
        weighted_impurity = 0
        total_instances = len(data)
        #print(total_instances)

        for unique_value in unique_values:
            partition = data[data[:, feat_index] == unique_value, :]
            #print(partition, unique_value)
            node = Node(data=partition)
            split_nodes[unique_value] = node
            partition_y = self.get_y(partition)
            node_impurity = self.calculate_impurity(partition_y)
            weighted_impurity += (len(partition) / total_instances) * node_impurity
        return split_nodes, weighted_impurity

    def best_split(self, node, depth=0):
        """
        Trouve le meilleur découpage pour un noeud donné.

        Arguments:
        node (Node) : Le noeud actuel à diviser.
        depth (int, optionnel) : La profondeur actuelle dans l'arbre (par défaut 0).
        """
        #we check if we reached max depth first if so then it's a leaf and we get the predicted class
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            y = self.get_y(node.data)
            node.pred_class = self.get_pred_class(y)
            return
        # we check if the criteria (entropy or gini) was met then same it's a leaf
        if self.meet_criteria(node):
            node.is_leaf = True
            y = self.get_y(node.data)
            node.pred_class = self.get_pred_class(y)
            return

        index_feature_split = -1
        min_impurity = float('inf')
        data = node.data

        for i in range(data.shape[1] - 1):
            split_nodes, weighted_impurity = self.split_on_feature(node.data, i)
            if weighted_impurity < min_impurity:
                child_nodes, min_impurity = split_nodes, weighted_impurity
                index_feature_split = i

        node.children = child_nodes
        node.split_on = index_feature_split

        for child_node in child_nodes.values():
            self.best_split(child_node, depth + 1)  

    def meet_criteria(self, node):
        """
        Vérifie si un noeud satisfait les critères d'impureté (impureté = 0).

        Arguments:
        node (Node) : Le noeud à vérifier.

        Retourne:
        bool : True si le critère est satisfait, sinon False.
        """
        y = self.get_y(node.data)
        return self.calculate_impurity(y) == 0

    def calculate_impurity(self, Y):
        """
        Calcule l'impureté d'un ensemble d'étiquettes en fonction du critère.

        Arguments:
        Y (ndarray) : Les étiquettes des données.

        Retourne:
        float : L'impureté calculée.
        """
        if self.criterion == 'entropy': #the higher the probability the lower the uncertinty  expected value of uncertinty 
            return self.calculate_entropy(Y)
        elif self.criterion == 'gini':
            return self.calculate_gini(Y)

    @staticmethod
    def calculate_entropy(Y):
        """
        Calcule l'entropie (mesure d'impureté) d'un ensemble d'étiquettes.

        Arguments:
        Y (ndarray) : Les étiquettes des données.

        Retourne:
        float : L'entropie calculée.
        """
        class_weights = {0: 1.3, 1: 1.0}
        labels, labels_counts = np.unique(Y, return_counts=True)
        weighted_counts = np.array([labels_counts[i] * class_weights[labels[i]] for i in range(len(labels))])
        labels_counts = weighted_counts
        #print(labels_counts)
        total_instances = len(Y)
        entropy = sum([label_count / total_instances * np.log2(1 / (label_count / total_instances)) for label_count in labels_counts])
        return entropy 
    
    @staticmethod
    def calculate_gini(Y):  #gini impurity is the probabily of having diffrent values 
        """
        Calcule l'impureté de Gini d'un ensemble d'étiquettes.

        Arguments:
        Y (ndarray) : Les étiquettes des données.

        Retourne:
        float : L'impureté de Gini calculée.
        """
        labels, labels_counts = np.unique(Y, return_counts=True)
        #print(Y)
        total_instances = len(Y)
        gini = 1 - sum((count / total_instances) ** 2 for count in labels_counts)
        #print('gini', gini)
        return gini
    
    @staticmethod
    def get_pred_class(Y):
        """
        Récupère la classe prédite d'un ensemble d'étiquettes en fonction du plus grand nombre d'occurrences.

        Arguments:
        Y (ndarray) : Les étiquettes des données.

        Retourne:
        int : La classe prédite.
        """
        labels, labels_counts = np.unique(Y, return_counts=True)
        index = np.argmax(labels_counts)
        #print("predicted label ", labels[index])
        return labels[index]

    @staticmethod
    def get_y(data):
        """
        Récupère les étiquettes (Y) des données.

        Arguments:
        data (ndarray) : Les données à partir desquelles extraire les étiquettes.

        Retourne:
        ndarray : Les étiquettes des données.
        """
        y = data[:, -1]
        return y
    
    def print_tree(self, node=None, depth=0):
        """
        Affiche l'arbre de décision de manière lisible.

        Arguments:
        node (Node, optionnel) : Le noeud à partir duquel afficher l'arbre (par défaut, la racine).
        depth (int, optionnel) : La profondeur du noeud (par défaut 0).
        """
        if node is None:
            node = self.root 

        indent = "    " * depth 

        if node.is_leaf:
            print(f"{indent}🌿 Leaf Node → Predicted class: {node.pred_class}")
            return

        print(f"{indent}🔹 Feature {node.split_on} (Splitting on this feature)")

        for value, child in node.children.items():
            print(f"{indent} ├── Value: {value}")
            self.print_tree(child, depth + 1)  


    def __str__ (self):
        return f"DecisionTree(depth={self.max_depth}, criterion={self.criterion})"
