from experimentations.experiment_manager import ExperimentManager
from model.classifiers.naive_bayes import NaiveBayesClassifier
from model.classifiers.perceptron import PerceptronClassifier
from model.classifiers.logistic_regression import LogisticRegressionClassifier
from model.classifiers.decision_tree import DecisionTreeClassifier
from model.classifiers.ada_boost import AdaBoost
from model.featurizers.count_vector import CountVectorFeaturizer
from model.featurizers.tfidf import TfidfFeaturizer
from model.preprocessors.tokenizer import Tokenizer
from model.preprocessors.lemmatizer import lemmatize
from model.pipeline.plagiarism_detector import PlagiarismDetector
from gui.config import *


class Model ():

    def __init__ (self):
        self.experiment_manager = ExperimentManager()
        self.detector_builder = DetectorBuilder()
        self.detectors_count = 0
        self._last_experiment = {
            "infos": [],
            "results": []
        }

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
        name = self.get_new_detector_name()
        self.experiment_manager.add_detector(detector, name)
        self._last_experiment["results"] = [self.experiment_manager.run_on_untrained()]
        self._last_experiment["infos"] = [(name, detector.__str__())]

    def get_last_experiment (self):
        return self._last_experiment



class DetectorBuilder ():

    def __init__ (self):
        self.attributes = {
            CLASSIFIER_REF: CLASSIFIER_DEFAULT,
            FEATURIZER_REF: FEATURIZER_DEFAULT,
            SWD_REF: SWD_DEFAULT,
            LEMMA_REF: LEMMA_DEFAULT,
            MAX_DEPTH_REF: MAX_DEPTH_DEFAULT,
            EPOCHS_REF: EPOCHS_DEFAULT,
            LEARNING_RATE_REF: LEARNING_RATE_DEFAULT,
            THRESHOLD_REF: THRESHOLD_DEFAULT,
            TOLERANCE_REF: TOLERANCE_DEFAULT,
            PATIENCE_REF: PATIENCE_DEFAULT,
            CRITERION_REF: CRITERION_DEFAULT,
            BAG_BOOST_REF: CRITERION_DEFAULT,
            N_ESTIMATORS_REF: N_ESTIMATORS_DEFAULT
        }
        intval = lambda x: self.is_number(x, int)
        floatval = lambda x: self.is_number(x, float)
        self.validators = {
            MAX_DEPTH_REF: intval,
            EPOCHS_REF: intval,
            LEARNING_RATE_REF: floatval,
            THRESHOLD_REF: floatval,
            TOLERANCE_REF: floatval,
            PATIENCE_REF: floatval,
            N_ESTIMATORS_REF: floatval
        }

    def is_number (self, obj, type_):
        try:
            return type_(obj)
        except ValueError:
            return False

    def set_ (self, attribute_ref, value):
        print(attribute_ref, value)
        if attribute_ref in self.validators:
            valid = self.validators[attribute_ref](value)
            print("valid", valid)
            if valid != False:
                self.attributes[attribute_ref] = valid
        elif attribute_ref in self.attributes:
            self.attributes[attribute_ref] = value

    def build (self):
        detector = PlagiarismDetector(self.build_classifier(), self.build_featurizer(), self.build_tokenizer())
        print(detector)
        return detector

    def build_tokenizer (self):
        stop_words, lemmatizer = None, None
        if self.attributes[SWD_REF]:
            stop_words = Tokenizer.DEFAULT_STOPWORDS
        if self.attributes[LEMMA_REF]:
            lemmatizer = lemmatize
        return Tokenizer(stop_words=stop_words, lemmatizer=lemmatizer)

    def build_featurizer (self):
        if self.attributes[FEATURIZER_REF] == TFIDF_DES:
            return TfidfFeaturizer()
        return CountVectorFeaturizer()

    def build_classifier (self):
        base_model = NaiveBayesClassifier()
        if self.attributes[CLASSIFIER_REF] == PERCEPTRON_DES:
            base_model = PerceptronClassifier(learning_rate=self.attributes[LEARNING_RATE_REF], epochs=self.attributes[EPOCHS_REF], patience=self.attributes[PATIENCE_REF], tolerance=self.attributes[TOLERANCE_REF])
        elif self.attributes[CLASSIFIER_REF] == LOGISTIC_DES:
            base_model = LogisticRegressionClassifier(learning_rate=self.attributes[LEARNING_RATE_REF], epochs=self.attributes[EPOCHS_REF], threshold=self.attributes[THRESHOLD_REF])
        elif self.attributes[CLASSIFIER_REF] == DECISION_TREE_DES:
            base_model = DecisionTreeClassifier(max_depth=self.attributes[MAX_DEPTH_REF], criterion=self.attributes[CRITERION_REF])
        if self.attributes[BAG_BOOST_REF] == ADA_BOOST_DES:
            return AdaBoost(base_model=base_model, models_count=self.attributes[N_ESTIMATORS_REF])
        if self.attributes[BAG_BOOST_REF] == BAGGING_DES:
            return AdaBoost(base_model=base_model, models_count=self.attributes[N_ESTIMATORS_REF])
        return base_model

    