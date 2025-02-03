import re
from nltk.corpus import stopwords


class Tokenizer:

    def tokenize(self, text):
        #pattens for respectively numbers, words and punctuation 
        token_pattern = r""" 
            \w+(?:[-'’]\w+)?
        """
        text = text.lower() #transforme all the text to lowercase
        pattern = re.compile(token_pattern, re.VERBOSE) #compile the regular expressions
        
        tokens = pattern.findall(text)  #extract all tokens from the input text that matches the pattern
        
        stop_words = set(stopwords.words('english'))
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
        return filtered_sentence



if __name__ == "__main__":

    tzer = Tokenizer()
    print(tzer.tokenize("Il s'agit de tokenizer cette phrase ; could you ?"))