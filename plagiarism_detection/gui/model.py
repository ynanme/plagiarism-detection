from experimentations.experiment_manager import ExperimentManager
from model.classifiers.naive_bayes import NaiveBayesClassifier
from model.classifiers.perceptron import PerceptronClassifier
from model.classifiers.logistic_regression import LogisticRegressionClassifier
from model.featurizers.count_vector import CountVectorFeaturizer
from model.featurizers.tfidf import TfidfFeaturizer
from model.tokenizers.tokenizer import Tokenizer
from model.tokenizers.lemmatizer import lemmatize
from model.pipeline.plagiarism_detector import PlagiarismDetector
from gui.config import *


class Model ():

    def __init__ (self):
        self.experiment_manager = ExperimentManager()
        self.detector_builder = DetectorBuilder()
        self.detectors_count = 0
        self._last_experiment = [], []

    def detector_is_being_built (self, attribute, value):
        self.detector_builder.set_(attribute, value)

    def gets_fed (self, data):
        self.experiment_manager.set_data(data)

    def get_new_detector_name (self):
        self.detectors_count += 1
        return self.detectors_count

    def add_detector (self):
        self.experiment_manager.add_detector(self.detector_builder.build(), self.get_new_detector_name())
        print(self.experiment_manager)

    def add_detector_to_experiments (self):
        detector = self.detector_builder.build()
        self.experiment_manager.add_detector(detector, self.get_new_detector_name())
        #self._last_experiment = self.experiment_manager.run_on_untrained()

    def get_last_experiment (self):
        return self._last_experiment



class DetectorBuilder ():

    def __init__ (self):
        self.attributes = {
            CLASSIFIER_REF: NAIVE_BAYES_DES,
            FEATURIZER_REF: COUNT_VECTOR_DES,
            SWD_REF: True,
            LEMMA_REF: False,
            EPOCHS_REF: 1000
        }

    def set_ (self, attribute_ref, value):
        print(attribute_ref, value)
        if attribute_ref in self.attributes:
            self.attributes[attribute_ref] = value

    def build (self):
        return PlagiarismDetector(self.build_classifier(), self.build_featurizer(), self.build_tokenizer())

    def build_tokenizer (self):
        print(self.attributes[SWD_REF], self.attributes[LEMMA_REF])
        stop_words, lemmatizer = None, None
        if self.attributes[SWD_REF]:
            stop_words = Tokenizer.DEFAULT_STOPWORDS
        if self.attributes[LEMMA_REF]:
            lemmatizer = lemmatize
        return Tokenizer(stop_words=stop_words, lemmatizer=lemmatizer)

    def build_featurizer (self):
        print(self.attributes[FEATURIZER_REF])
        if self.attributes[FEATURIZER_REF] == TFIDF_DES:
            return TfidfFeaturizer()
        return CountVectorFeaturizer()

    def build_classifier (self):
        print(self.attributes[CLASSIFIER_REF])
        if self.attributes[CLASSIFIER_REF] == PERCEPTRON_DES:
            return PerceptronClassifier(epochs=self.attributes[EPOCHS_REF])
        if self.attributes[CLASSIFIER_REF] == LOGISTIC_DES:
            return LogisticRegressionClassifier(epochs=self.attributes[EPOCHS_REF])
        return NaiveBayesClassifier()

    