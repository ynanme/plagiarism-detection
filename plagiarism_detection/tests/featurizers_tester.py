from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from model.featurizers.count_vector_featurizer import CountVectorFeaturizer
from model.featurizers.tfidf_featurizer import TfidfFeaturizer
from model.preprocessors.tokenizer import Tokenizer
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


def test_transform (our_featurizer, their_featurizer):
    print(f"[Tests] [{type(our_featurizer).__name__}::transform] launched")
    texts = read_files(DIRECTORY)
    our_featurizer.fit(texts)
    their_featurizer.fit(texts)
    if not (our_featurizer.transform(texts).all() == their_featurizer.transform(texts).toarray().all()):
        compare_vocabularies(our_vocabulary, their_vocabulary)
        print(f"[Tests] [{type(our_featurizer).__name__}::transform] failed")
        return False
    print(f"[Tests] [{type(our_featurizer).__name__}::transform] passed")
    return True


def compare_vocabularies (ours, theirs):
    print()
    print(f"lengths     ours={len(ours)} theirs={len(theirs)}")
    print("not in ours   ", [token for token in theirs.keys() if token not in ours.keys()])
    print("not in theirs ", [token for token in ours.keys() if token not in theirs.keys()])
    print()



if __name__ == "__main__":

    cv = CountVectorFeaturizer()
    tfidf = TfidfFeaturizer()

    cv_sk = CountVectorizer(tokenizer=Tokenizer().tokenize)
    tfidf_sk = TfidfVectorizer(tokenizer=Tokenizer().tokenize)

    test_fit(cv, cv_sk)
    test_fit(tfidf, tfidf_sk)

    test_transform(cv, cv_sk)
    test_transform(tfidf, tfidf_sk)

    """
    f = CountVectorizer(tokenizer=Tokenizer().tokenize)
    f.fit(["chat noir", "chat blanc"])
    print(f.vocabulary_)
    f.fit(["sur le toît"])
    print(f.vocabulary_)
    """