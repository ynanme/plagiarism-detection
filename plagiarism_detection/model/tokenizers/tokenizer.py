from nltk.corpus import stopwords
import re
from typing import List
from charset_normalizer import from_path


# In[2]:
class Tokenizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def __init__ (self, stop_words=None, lemmatizer=None):
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer

    def tokenize(self, text):
        #pattens for respectively numbers, words and punctuation 
        token_pattern = r""" 
            \w+(?:[-'’]\w+)?
        """
        text = text.lower() #transforme all the text to lowercase
        pattern = re.compile(token_pattern, re.VERBOSE) #compile the regular expressions
        tokens = pattern.findall(text)  #extract all tokens from the input text that matches the pattern
        
        stop_words = set(stopwords.words('english')) if self.stop_words is None else self.stop_words
        filtered_sentence = []
        
        for w in tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        """
        print("tokens",tokens)
        print("\n")
        print("tokens without stopwords",filtered_sentence)
        print("and" in filtered_sentence)
        """
        if self.lemmatizer is not None:
            return self.lemmatizer(filtered_sentence)
        return filtered_sentence

        return filtered_tokens
    
    @staticmethod
    def read_file(file_path):
        #TODO: fix CharsetMatch problem
        result = from_path(file_path).best() #detects best encoding for the file
        if result:
            return str(result)  
        else:
            raise ValueError(f"Could not detect encoding for file: {file_path}")
    
    @staticmethod
    def generate_ngrams(tokens, ngram):

        ngrams = []
        for n in range(ngram[0], ngram[1] + 1):
            ngrams += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            #print(ngrams)
        return ngrams
