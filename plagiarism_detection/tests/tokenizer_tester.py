"""
exécuter depuis la racine du projet (plagiarism_detection)
python3 -m tests.tokenizer_tester
"""


import nltk
from nltk.tokenize import word_tokenize

from util.util import read_files
from model.tokenizers.tokenizer import Tokenizer


DIRECTORY = "data/1"
TEXTS = read_files(DIRECTORY)
OUR_TOKENIZER = Tokenizer()

for text in TEXTS:
    assert OUR_TOKENIZER.tokenize(text) == word_tokenize(text)