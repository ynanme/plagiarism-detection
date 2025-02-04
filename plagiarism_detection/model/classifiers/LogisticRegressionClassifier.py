#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np


# In[15]:


class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, matrix, binary_labels):
        nb_docs, nb_vocab = matrix.shape
        self.weights = np.zeros(nb_vocab)
        self.bias = 0

        for epoch in range(self.epochs):
            z = np.dot(matrix, self.weights) + self.bias
            predictions = self.sigmoid(z)
            #loss = -np.mean(binary_labels * np.log(predictions) + (1 - binary_labels) * np.log(1 - predictions))
            #print(f"Epoch {epoch}: Loss = {loss}")  
            dw = np.dot(matrix.T, (predictions - binary_labels))
            db = predictions - binary_labels

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, matrix):
        z = np.dot(matrix, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        return (probabilities >= 0.5).astype(int)


# In[11]:


"""folder_path = "../data-plagiarism"

# Load labels
filtered_labels = load_labels(folder_path)
documents = [Tokenizer.read_file(os.path.join(folder_path, file)) for file in filtered_labels.keys()]#read all documents

# Test both featurizers
for featurizer_class in [TfidfFeaturizer]:
    featurizer = featurizer_class()
    featurizer.fit(documents)
    matrix = featurizer.transform(documents)
    #print("matrix", matrix)
    #word_probabilities = compute_word_probabilities(filtered_labels, featurizer, matrix)
    classifier = LogisticRegressionClassifier()
    binary_labels = np.array([1 if label == "plagiat" else 0 for label in filtered_labels.values()])
    #print("binary labels", binary_labels)
    #print("matrix", filtered_labels)
    classifier.fit(matrix, binary_labels)
    
    # Classify files
    results = []
    for file_name, feature_vector in zip(filtered_labels.keys(), matrix):
        prediction = classifier.predict(feature_vector.reshape(1, -1))[0]
        #print("prediction", prediction)
        results.append({
            "File Name": file_name,
            "Expected Label": filtered_labels[file_name],
            "Predicted Label": "plagiat" if prediction == 1 else "non_plagiat"
        })

    df_results = pd.DataFrame(results)
    df_results["Mismatch"] = df_results["Expected Label"] != df_results["Predicted Label"]

    print(f"Results for {featurizer_class.__name__}:")
    #print(df_results.to_string(index=False))

    mismatches = df_results[df_results["Mismatch"]]
    print(f"Mismatches for {featurizer_class.__name__}:", len(mismatches))
    #print(mismatches.to_string(index=False))
    
"""


# In[ ]:


"""for file_name in filtered_labels.keys():
    content = Tokenizer.read_file(os.path.join(folder_path, file_name))
    # Transform the individual document
    transformed_content = featurizer.transform([content])  # A single document
    prediction = classifier.predict(transformed_content)[0]  # Get the predicted label for this document
    results.append({
        "File Name": file_name,
        "Expected Label": filtered_labels[file_name],
        "Predicted Label": prediction
    })"""


# In[ ]:




