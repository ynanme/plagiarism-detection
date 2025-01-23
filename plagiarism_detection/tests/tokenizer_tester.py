"""
exécuter depuis la racine du projet (plagiarism_detection)
python3 -m tests.tokenizer_tester
"""


import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

from util.util import read_files
from model.tokenizers.tokenizer import Tokenizer


DIRECTORY = "data/1"
OUR_TOKENIZER = Tokenizer()
THEIR_TOKENIZER = CountVectorizer(stop_words=stopwords.words('english')).build_tokenizer()


def test_tokenize ():
    print("[Tests] [Tokenizer::tokenize] [launched]")
    texts = read_files(DIRECTORY)
    for text in texts:
        ours = OUR_TOKENIZER.tokenize(text)
        theirs = [token.lower() for token in THEIR_TOKENIZER(text)]
        if ours != theirs:
            compare_tokens_results(ours, theirs)
            compare_tokens_results_too(ours, theirs)
            print("[Tests] [Tokenizer::tokenize] [failed]")
            return False
    print("[Tests] [Tokenizer::tokenize] [passed]")
    return True


def compare_tokens_results (ours, theirs):
    print()
    print(f"lengths     ours={len(ours)} theirs={len(theirs)}")
    print("not in ours   ", [token for token in theirs if token not in ours])
    print("not in theirs ", [token for token in ours if token not in theirs])
    print()


def compare_tokens_results_too (ours, theirs):
    print()
    print("tokenize computed")
    print(ours)
    print("when nltk computed")
    print(theirs)
    print()



if __name__ == "__main__":

    print('a' in stopwords.words('english'))
    print('is' in stopwords.words('english'))
    test_tokenize()