import spacy


nlp = spacy.load("en_core_web_sm")

def lemmatize (words):
    doc = nlp(" ".join(words))
    return [token.lemma_ for token in doc]