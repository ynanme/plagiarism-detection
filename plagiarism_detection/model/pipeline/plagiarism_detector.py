class PlagiarismDetector:

    def __init__ (self, classifier, featurizer):
        self.tokenizer = Tokenizer()
        self.classifier = classifier
        self.featurizer = featurizer

    def classify (text):
        tokens = self.tokenizer.tokenize(text)
        features = self.featurizer.fit_transform(tokens)
        classification = self.classifier.predict(features)
        return self.interpret(classification)

    def interpret (classification):
        return "no plagiarism" if classification == 0 else "plagiarism"