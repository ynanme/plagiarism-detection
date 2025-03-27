import os
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split

from util.util import read_file
from experimentations.experiments_utils import *

from model.preprocessors.tokenizer import Tokenizer

from model.featurizers.count_vector import CountVectorFeaturizer
from model.featurizers.tfidf import TfidfFeaturizer

from model.classifiers.naive_bayes import NaiveBayesClassifier
from model.classifiers.logistic_regression import LogisticRegressionClassifier 
from model.classifiers.perceptron import PerceptronClassifier
from model.classifiers.decision_tree import DecisionTreeClassifier
#from model.classifiers.RandomForestClassifier import RandomForestClassifier


def map_value(value: str) -> str:
    return 'plagiat' if value in ['heavy', 'cut', 'light'] else 'non_plagiat'

def load_labels(folder_path: str, file_name: str) -> Dict[str, str]:
    expected_results = pd.read_csv(os.path.join(folder_path, file_name), sep=",")
    labels = dict(zip(expected_results["File"], expected_results['Category']))
    return {k: map_value(v) for k, v in sorted(labels.items())}

hyperparameters = {
    (CountVectorFeaturizer, NaiveBayesClassifier): {},
    (CountVectorFeaturizer, LogisticRegressionClassifier): {'learning_rate': 0.05, 'epochs': 100, 'threshold': 0.5, 'patience': 10, 'tolerance': 1e-4},
    (CountVectorFeaturizer, PerceptronClassifier): {'learning_rate': 0.001, 'epochs': 1000, 'patience': 10, 'tolerance': 1e-4},
    #(CountVectorFeaturizer, DecisionTreeClassifier): {'max_depth': 5, 'criterion': 'gini'},
    (CountVectorFeaturizer, DecisionTreeClassifier): {'max_depth': 10, 'criterion': 'entropy'},
    #(CountVectorFeaturizer, RandomForestClassifier): {},
    
    (TfidfFeaturizer, NaiveBayesClassifier): {}, 
    (TfidfFeaturizer, LogisticRegressionClassifier): {'learning_rate': 0.01, 'epochs': 100, 'threshold': 0.56, 'patience': 10, 'tolerance': 1e-4},
    (TfidfFeaturizer, PerceptronClassifier): {'learning_rate': 0.0001, 'epochs': 700, 'patience': 10, 'tolerance': 1e-4},
    #(TfidfFeaturizer, DecisionTreeClassifier): {'max_depth': 2, 'criterion': 'gini'}, #always at 16 no matter the depth
    (TfidfFeaturizer, DecisionTreeClassifier): {'max_depth': 10, 'criterion': 'entropy'}, #always at 16 no matter the depth
    #(TfidfFeaturizer, RandomForestClassifier): {},
}
Fhyperparameters = {
    CountVectorFeaturizer: {'ngram': [1, 1]},
    TfidfFeaturizer: {'ngram': [1, 1]}
}

if __name__ == "__main__":
    folder_path = "data/1"

    filtered_labels = load_labels(folder_path, "file_information.csv")
    labels_df = pd.DataFrame(list(filtered_labels.items()), columns=['File', 'label'])

    plagiat_df = labels_df[labels_df['label'] == 'plagiat']
    non_plagiat_df = labels_df[labels_df['label'] == 'non_plagiat']

    test_size = 0.25
    plagiat_test_size = int(len(plagiat_df) * test_size)
    non_plagiat_test_size = int(len(non_plagiat_df) * test_size)

    plagiat_train, plagiat_test = train_test_split(
        plagiat_df, test_size=plagiat_test_size, random_state=42
    )

    non_plagiat_train, non_plagiat_test = train_test_split(
        non_plagiat_df, test_size=non_plagiat_test_size, random_state=42
    )

    train_df = pd.concat([plagiat_train, non_plagiat_train])
    test_df = pd.concat([plagiat_test, non_plagiat_test])

    train_labels = dict(zip(train_df['File'], train_df['label']))
    test_labels = dict(zip(test_df['File'], test_df['label']))

    print("Test plagiat count:", len(plagiat_test))
    print("Test non-plagiat count:", len(non_plagiat_test))
    print("Train plagiat count:", len(plagiat_train))
    print("Train non-plagiat count:", len(non_plagiat_train))
    
    documents = [read_file(os.path.join(folder_path, file)) for file in train_labels.keys()]
    test_documents = [read_file(os.path.join(folder_path, file)) for file in test_labels.keys()]
    
    for featurizer_class in [CountVectorFeaturizer, TfidfFeaturizer]:
        featurizer_hyperparameters = Fhyperparameters[featurizer_class]
        featurizer = featurizer_class(**featurizer_hyperparameters)  # Pass hyperparameters

        featurizer.fit(documents)
        matrix = featurizer.transform(documents)
        test_matrix = featurizer.transform(test_documents)
        
        for classifier_class in [NaiveBayesClassifier, LogisticRegressionClassifier, PerceptronClassifier, DecisionTreeClassifier]:
            classifier_hyperparameters = hyperparameters[(featurizer_class, classifier_class)]
            classifier = classifier_class(**classifier_hyperparameters)  # Pass hyperparameters


            binary_labels = (
                np.array([1 if label == "plagiat" else 0 for label in train_labels.values()])
                if classifier_class in [LogisticRegressionClassifier, NaiveBayesClassifier, DecisionTreeClassifier]
                else [1 if label == "plagiat" else -1 for label in train_labels.values()]
            )
            classifier.fit(matrix, binary_labels)
            
            results = []
            for file_name, feature_vector in zip(test_labels.keys(), test_matrix):
                prediction = classifier.predict([feature_vector])[0]
                #print(classifier_class, featurizer_class, prediction)
                predicted_label = "plagiat" if prediction == 1 else "non_plagiat"
                results.append({
                    "File Name": file_name,
                    "Expected Label": str(filtered_labels[file_name]),
                    "Predicted Label": str(predicted_label)
                })
            
            df_results = pd.DataFrame(results)
            df_results["Mismatch"] = df_results["Expected Label"] != df_results["Predicted Label"]
            
            #print(f"\nResults for {featurizer_class.__name__} with {classifier_class.__name__}:")
            print(df_results.to_string(index=False))
            
            mismatches = df_results[df_results["Mismatch"]]
            print(f"Mismatches for {featurizer_class.__name__} with {classifier_class.__name__}: {len(mismatches)}, {100-len(mismatches)*100/24}\n")
            #print(mismatches.to_string(index=False))
