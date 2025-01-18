import nltk
from nltk.tokenize import word_tokenize

import util
from tokenizer import Tokenizer


DIRECTORY = "../data/1"
TEXTS = read_files(DIRECTORY)
OUR_TOKENIZER = Tokenizer()

for text in TEXTS:
    assert OUR_TOKENIZER.tokenize(text) == word_tokenize(text)