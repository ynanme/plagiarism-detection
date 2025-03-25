import numpy as np

class Node:
    def __init__(self, data=None, children=None, split_on = None, pred_class=None, is_leaf=False):

        self.data = data
        self.children = children
        self.split_on = split_on
        self.pred_class = pred_class
        self.is_leaf = is_leaf

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, criterion='entropy'):
        self.root = Node()
        self.max_depth = max_depth
        self.criterion = criterion
    
    def fit(self, X, Y):
        data = np.column_stack([X, Y])
        self.root.data = data
        #print(np.unique(Y, return_counts=True))
        self.best_split(self.root)

    def predict(self, X):
        #print("in decision", len(X), len(X[0]))
        predictions = np.array([self.traverse_tree(x, self.root) for x in X])
        #self.print_tree()
        #print(predictions)
        return predictions

    def traverse_tree(self, x, node):
        if node.is_leaf:
            #print(node.pred_class)
            return node.pred_class

        feat_value = x[node.split_on]
        if feat_value not in node.children:  
            return node.pred_class
        
        predicted_class = self.traverse_tree(x, node.children[feat_value])

        return predicted_class

    def split_on_feature(self, data, feat_index):
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
        y = self.get_y(node.data)
        return self.calculate_impurity(y) == 0

    def calculate_impurity(self, Y):
        if self.criterion == 'entropy': #the higher the probability the lower the uncertinty  expected value of uncertinty 
            return self.calculate_entropy(Y)
        elif self.criterion == 'gini':
            return self.calculate_gini(Y)

    @staticmethod
    def calculate_entropy(Y):
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
        labels, labels_counts = np.unique(Y, return_counts=True)
        #print(Y)
        total_instances = len(Y)
        gini = 1 - sum((count / total_instances) ** 2 for count in labels_counts)
        #print('gini', gini)
        return gini
    
    @staticmethod
    def get_pred_class(Y):
        labels, labels_counts = np.unique(Y, return_counts=True)
        index = np.argmax(labels_counts)
        #print("predicted label ", labels[index])
        return labels[index]

    @staticmethod
    def get_y(data):
        y = data[:, -1]
        return y
    
    def print_tree(self, node=None, depth=0):
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
