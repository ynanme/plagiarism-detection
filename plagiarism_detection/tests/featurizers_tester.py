from sklearn.feature_extraction.text import CountVectorizer

from model.featurizers.count_vector_featurizer import CountVectorFeaturizer
from model.tokenizers.tokenizer import Tokenizer
from util.util import read_files



DIRECTORY = "data/1"


def test_fit (our_featurizer, their_featurizer):
    print(f"[Tests] [{type(our_featurizer).__name__}::fit] launched")
    texts = read_files(DIRECTORY)
    our_featurizer.fit(texts)
    their_featurizer.fit(texts)
    our_vocabulary = our_featurizer.vocabulary
    their_vocabulary = their_featurizer.vocabulary_
    if not (set(our_featurizer.vocabulary.keys()) == set(their_featurizer.vocabulary_.keys())):
        compare_vocabularies(our_vocabulary, their_vocabulary)
        print(f"[Tests] [{type(our_featurizer).__name__}::fit] failed")
        return False
    print(f"[Tests] [{type(our_featurizer).__name__}::fit] passed")
    #print(our_vocabulary, their_vocabulary)
    return True


def compare_vocabularies (ours, theirs):
    print()
    print(f"lengths     ours={len(ours)} theirs={len(theirs)}")
    print("not in ours   ", [token for token in theirs.keys() if token not in ours.keys()])
    print("not in theirs ", [token for token in ours.keys() if token not in theirs.keys()])
    print()



if __name__ == "__main__":

    test_fit(CountVectorFeaturizer(), CountVectorizer(tokenizer=Tokenizer().tokenize))