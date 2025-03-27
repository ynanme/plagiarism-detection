from nltk.corpus import stopwords
import re
from typing import List, Tuple
from charset_normalizer import from_path


class Tokenizer:

    """
    Une classe pour tokenizer du texte, supprimer les mots vides (stopwords),
    et générer des n-grams. Elle supporte également une lemmatisation optionnelle.

    Attributs :
        stop_words (set): Un ensemble de mots vides à filtrer parmi les tokens.
        lemmatizer (callable, optionnel): Une fonction qui applique la lemmatisation sur les tokens.
    """

    DEFAULT_STOPWORDS = set(stopwords.words('english'))

    def __init__(self, stop_words=DEFAULT_STOPWORDS, lemmatizer=None):
        """
        Initialise le tokenizer avec un filtrage des mots vides et une lemmatisation optionnelle.

        Args :
            stop_words : Un ensemble de mots vides personnalisé. Par défaut, c'est l'ensemble des mots vides en anglais.
            lemmatizer : Fonction qui applique la lemmatisation aux tokens.
        """
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer

    def tokenize(self, text) -> List[str]:
        """
        Tokenise le texte d'entrée en mots, supprime les mots vides et applique une lemmatisation optionnelle.

        Args :
            text (str) : Le texte à tokeniser.

        Retourne :
            List[str] : Une liste de tokens filtrés.
        """
        #pattens for respectively numbers, words and punctuation 
        token_pattern = r""" \w+(?:[-'’]\w+)?"""
        text = text.lower() #transforme all the text to lowercase
        pattern = re.compile(token_pattern, re.VERBOSE) #compile the regular expressions
        tokens = pattern.findall(text)  #extract all tokens from the input text that matches the pattern
        
        stop_words = set(stopwords.words('english')) if self.stop_words is None else self.stop_words
        filtered_sentence = []
        
        for w in tokens:
            if w not in stop_words:
                filtered_sentence.append(w)

        if self.lemmatizer is not None:
            return self.lemmatizer(filtered_sentence)
        return filtered_sentence

    @staticmethod
    def read_file(file_path) -> str:
        """
        Lit un fichier texte en détectant automatiquement son encodage.

        Args :
            file_path (str) : Le chemin du fichier à lire.

        Retourne :
            str : Le contenu du fichier sous forme de chaîne.

        Lève :
            ValueError : Si l'encodage du fichier ne peut pas être détecté.
        """
        result = from_path(file_path).best() #detects best encoding for the file
        if result:
            return str(result)  
        else:
            raise ValueError(f"Could not detect encoding for file: {file_path}")
    
    @staticmethod
    def generate_ngrams(tokens: List[str], ngram: Tuple[int, int]) -> List[str]:
        """
        Génère des n-grams à partir d'une liste de tokens.

        Args :
            tokens (List[str]) : Une liste de tokens.
            ngram (Tuple[int, int]) : Un tuple spécifiant la longueur minimale et maximale des n-grams.

        Retourne :
            List[str] : Une liste de n-grams.
        """
        ngrams = []
        for n in range(ngram[0], ngram[1] + 1):
            ngrams += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            #print(ngrams)
        return ngrams


    def __str__(self) -> str:
            """
            Retourne une représentation sous forme de chaîne de l'instance de tokenizer.

            Retourne :
                str : Une chaîne formatée décrivant les paramètres du tokenizer.
            """
            return f"TOKENIZER(suppression_des_mots_vides={self.stop_words is not None}, lemmatisation={self.lemmatizer is not None})"