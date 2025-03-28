import numpy as np

from model.featurizers.featurizer import Featurizer


class PlagiarismDetector:

    def __init__ (self, classifier, featurizer, tokenizer):
        self.tokenizer = tokenizer
        self.featurizer = featurizer
        self.classifier = classifier
        if isinstance(self.featurizer, Featurizer):
            self.featurizer.set_tokenizer(self.tokenizer)
    
    def train (self, texts, categories):
        self.classifier.fit(self.featurizer.fit_transform(texts), np.array(categories))

    
    def classify (self, text):
        return self.classifier.predict(self.featurizer.transform([text]))[0]


    def interpret (self, classification):
        return "no plagiarism" if classification == 0 else "plagiarism"


    def __str__ (self):
        return f"PLAGIARISM DETECTION [{self.classifier}, {self.featurizer}, {self.tokenizer}]"

