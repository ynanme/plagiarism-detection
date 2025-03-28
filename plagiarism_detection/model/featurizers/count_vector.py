
import numpy as np
from collections import Counter
from typing import List
from model.preprocessors.tokenizer import Tokenizer
from model.featurizers.featurizer import Featurizer


class CountVectorFeaturizer (Featurizer):
    """
    La classe `CountVectorFeaturizer` permet de convertir un ensemble de documents
    texte en une matrice de caractéristiques basée sur des n-grammes.
    Elle utilise un tokenizer pour extraire les tokens et crée un vecteur de comptage des n-grammes.
    """
    
    def __init__(self, ngram=[1, 1]):
        """
        Initialise un modèle `CountVectorFeaturizer` avec des n-grammes.

        Args:
            ngram (list, optionnel): Plage de taille des n-grammes (par défaut [1, 1], donc unigramme).
        """
        super().__init__(Tokenizer(), ngram)
        self.tokenized_docs = [] 

    def fit(self, documents: List[str]) -> None:
        """
        Apprend un vocabulaire à partir des documents fournis et construit un vocabulaire basé sur les n-grammes.

        Args:
            documents (List[str]): Liste des documents texte à analyser.
        """
        token_counter = Counter()  
        self.tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents] 
        
        for tokens in self.tokenized_docs:
            ngrams = Tokenizer.generate_ngrams(tokens=tokens, ngram=self.ngram)
            token_counter.update(set(ngrams))
        
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(token_counter.most_common())}

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transforme une liste de documents texte en une matrice de comptage des n-grammes.

        Args:
            documents (List[str]): Liste des documents texte à transformer.

        Returns:
            np.ndarray: Matrice où chaque ligne représente un document et chaque colonne un n-gramme du vocabulaire.
        """
        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]  
        count_matrix = np.zeros((len(documents), len(self.vocabulary)), dtype=int)

        for i, tokens in enumerate(tokenized_docs):
            ngrams = Tokenizer.generate_ngrams(tokens=tokens, ngram=self.ngram)
            for token in ngrams:
                idx = self.vocabulary.get(token)
                if idx is not None:
                    count_matrix[i, idx] += 1
        
        return count_matrix

    
    def __str__ (self):
        return "CountVector"