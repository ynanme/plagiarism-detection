import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from model.tokenizers.tokenizer import Tokenizer
from model.featurizers.count_vector import CountVectorFeaturizer
from model.classifiers.naive_bayes import NaiveBayesClassifier


class PlagiarismDetector:

    def __init__ (self, classifier, featurizer):
        self.tokenizer = Tokenizer()
        self.classifier = classifier
        self.featurizer = featurizer
        self.set_tools()

    
    def train (self, texts, categories):
        #self.featurizer.fit(texts)
        #print(np.array(list(categories.values())))
        self.classifier.fit(self.featurizer.fit_transform(texts), np.array(categories))


    
    def classify (self, text):
        return self.classifier.predict(self.featurizer.transform([text]))

    def evaluate (self, texts, categories):
        y_computed = [self.classify(text) for text in texts]
        print("y", y_computed)
        corrects = sum(1 for computed, expected in zip(y_computed, categories) if computed == expected)
        return corrects / len(categories) if len(categories) > 0 else 1


    def interpret (self, classification):
        return "no plagiarism" if classification == 0 else "plagiarism"

    def set_tools (self):
        #self.classifier.init_class_priors(2)
        pass



if __name__ == "__main__":

    texts_for_train = [
        "the quick brown fox jumps over the lazy dog",
        "a fast brown fox leaps over a sleepy dog",
        "the quick brown fox is the best",
        "copy the quick brown fox over the lazy dog",
        "the clever fox avoided the hunter",
        "the lazy dog was caught sleeping"
    ]
    categories_for_train = {0:"non_plagiat", 1:"non_plagiat", 2:"plagiat", 3:"plagiat", 4:"non_plagiat", 5:"plagiat"}

    texts_for_test = [
        "the fox was quick to jump over the dog",
        "the lazy dog sleeps while the fox runs",
        "a clever fox jumps over a sleeping dog",
        "copy the text of the quick brown fox"
    ]
    categories_for_test = {0:"non_plagiat", 1:"plagiat", 2:"non_plagiat", 3:"plagiat"}

    inspector = PlagiarismDetector(NaiveBayesClassifier(), CountVectorFeaturizer())
    director = PlagiarismDetector(MultinomialNB(), CountVectorizer())

    inspector.train(texts_for_train, categories_for_train)
    director.trainn(texts_for_train, categories_for_train)

    print([inspector.classify(text) for text in texts_for_test])
    print([director.classifyy(text) for text in texts_for_test])

    print(inspector.evaluate(texts_for_test, categories_for_test))
    print(director.evaluatee(texts_for_test, categories_for_test))