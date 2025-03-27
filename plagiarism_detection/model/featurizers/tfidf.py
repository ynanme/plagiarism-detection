import numpy as np
from typing import List
from collections import Counter
from model.preprocessors.tokenizer import Tokenizer
from model.featurizers.featurizer import Featurizer

# Term freaquency: frequency of all words within a document
# Inverse document Frequency: inverse of how often a word appears accross al documents 
class TfidfFeaturizer (Featurizer):
    """
    La classe `TfidfFeaturizer` permet de convertir un ensemble de documents texte en une matrice TF-IDF.
    Elle utilise un tokenizer pour extraire les n-grammes des documents et calcule les valeurs TF-IDF pour chaque n-gramme.
    """

    def __init__(self, ngram=[1, 1]):
        """
        Initialise le modèle TF-IDF avec des n-grammes.

        Args:
            ngram (list, optionnel) : Plage de taille des n-grammes (par défaut [1, 1], donc unigramme).
        """
        super().__init__(Tokenizer(), ngram)  
        #self.tokenized_docs = []  
        self.idf_values = {}

    def fit(self, documents: List[str]) -> None:
        """
        Apprend le vocabulaire et calcule les valeurs IDF à partir des documents fournis.

        Args:
            documents (List[str]): Liste des documents texte à analyser.
        """
        nb_doc = len(documents) 
        token_counter = Counter()  

        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]
        
        for tokens in tokenized_docs:
            ngrams = Tokenizer.generate_ngrams(tokens=tokens, ngram=self.ngram)
            token_counter.update(set(ngrams)) #we only count each occurence of a token in a doc once, in other terms the number of times a token appears in all docs
        
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(token_counter.most_common())}

        #Calculate the idf values of each token, log of the number of docs devided by the number of docs that contain the term (+1 to avoid 0 values)
        self.idf_values = {
            token: np.log((nb_doc) / (1 + doc_freq)) + 1
            for token, doc_freq in token_counter.items()
        }

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transforme une liste de documents texte en une matrice TF-IDF.

        Args:
            documents (List[str]): Liste des documents texte à transformer.

        Returns:
            np.ndarray: Matrice où chaque ligne représente un document et chaque colonne un n-gramme du vocabulaire.
        """
        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]  # Tokenize input documents
        tfidf_matrix = np.zeros((len(documents), len(self.vocabulary)))

        #for each token:term of each document we calculate tf*idf
        for i, tokens in enumerate(tokenized_docs):
            ngrams = Tokenizer.generate_ngrams(tokens=tokens, ngram=self.ngram)
            term_counts = Counter(ngrams) #transformation to a counter
            total_terms = len(ngrams)

            for token, count in term_counts.items(): 
                tf = count / total_terms  # Term frequency
                idf = self.idf_values.get(token, 0)  
                if token in self.vocabulary:
                     tfidf_matrix[i, self.vocabulary[token]] = tf * idf
        return tfidf_matrix

