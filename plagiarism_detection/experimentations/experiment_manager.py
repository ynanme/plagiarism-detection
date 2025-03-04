import numpy as np

from util.util import partition, draw_plot, file_name


class ExperimentManager ():

    def __init__ (self, detectors, names, data=[[],[]], saving_results_path=None):
        self.detectors = {name: detector for name, detector in zip(names, detectors)}
        self.texts, self.categories = data
        self.saving_results_path = saving_results_path
        if not self.saving_results_path.endswith("/"):
            self.saving_results_path += "/"

    def train_detectors (self, texts, categories):
        #categories = {index: category for index,category in enumerate(categories)}
        for detector in self.detectors.values():
            detector.train(texts, categories)
    
    def evaluate_detectors (self, texts, categories):
        #categories = {index: category for index,category in enumerate(categories)}
        accuracies = []
        for detector in self.detectors.values():
            accuracies.append(detector.evaluate(texts, categories))
        return accuracies

    def get_indexes (self, count=None):
        count = len(self.texts) if count is None else count
        return [index for index in range(count)]

    def filter_indexes (self, indexes, filterr):
        if filterr is not None:
            return [index for index in indexes if self.categories[index] == filterr]
        return indexes

    def get_filters_description (self, filter_on_training, filter_on_testing):
        filters_description = ""
        filters_description += f"trained on {filter_on_training} " if filter_on_training is not None else ""
        filters_description += f"tested on {filter_on_testing}" if filter_on_testing is not None else ""
        return filters_description
    
    def get_names_extension (self):
        return "&".join(self.detectors.keys())

    
    def run_experiment (self, training_indexes, testing_indexes=None, repeat=1, description="Experiment"):
        run_on_trained = testing_indexes is None
        testing_texts = [] if run_on_trained else [self.texts[index] for index in testing_indexes]
        testing_texts_categories = [] if run_on_trained else [self.categories[index] for index in testing_indexes]
        all_accuracies = []
        for _ in range(repeat):
            accuracies = []
            trained_on_texts, trained_on_texts_categories = [], []
            for index in training_indexes:
                trained_on_texts.append(self.texts[index])
                trained_on_texts_categories.append(self.categories[index])
                self.train_detectors(trained_on_texts, trained_on_texts_categories)
                accuracies.append(self.evaluate_detectors(
                    trained_on_texts if run_on_trained else testing_texts,
                    trained_on_texts_categories if run_on_trained else testing_texts_categories
                ))
            all_accuracies.append(accuracies)
        accuracies = np.mean(np.array(all_accuracies), axis=0)
        draw_plot(
            [x for x in range(len(training_indexes))],
            [[accuracy[index] for accuracy in accuracies] for index in range(len(self.detectors))],
            self.detectors.keys(), ["Amount of data trained on", "Detector accuracy"], f"{description}",
            True, f"{self.saving_results_path}{file_name(description)}_{self.get_names_extension()}.png" if self.saving_results_path is not None else None
        )
        return accuracies

    def run_on_all (self, training_data_weight=1, filter_on_training=None, filter_on_testing=None, repeat=1):
        all_indexes = self.get_indexes()
        training_indexes = partition(all_indexes, [training_data_weight])[0]
        return self.run_experiment(
            self.filter_indexes(training_indexes, filter_on_training),
            self.filter_indexes(all_indexes, filter_on_testing),
            repeat,
            f"Evaluating on all data, training data is {round(training_data_weight*100)}%, {self.get_filters_description(filter_on_training, filter_on_testing)}, average on {repeat} repeats"
        )

    def run_on_trained (self, training_data_weight=1, filter_on_training=None, filter_on_testing=None, repeat=1):
        training_indexes = partition(self.get_indexes(), [training_data_weight])[0]
        #return self.run_experiment(training_indexes, description=f"Evaluating on trained data, training data is {round(training_data_weight*100)}%")
        return self.run_experiment(
            self.filter_indexes(training_indexes, filter_on_training),
            repeat=repeat,
            description=f"Evaluating on trained data, training data is {round(training_data_weight*100)}%, {self.get_filters_description(filter_on_training, filter_on_testing)}, average on {repeat} repeats"
        )

    def run_on_untrained (self, training_data_weight=0.8, filter_on_training=None, filter_on_testing=None, repeat=1):
        training_indexes, testing_indexes = partition(self.get_indexes(), [training_data_weight])
        #return self.run_experiment(training_indexes, testing_indexes, f"Evaluating on untrained data, training data is {round(training_data_weight*100)}%")
        return self.run_experiment(
            self.filter_indexes(training_indexes, filter_on_training),
            self.filter_indexes(testing_indexes, filter_on_testing),
            repeat,
            f"Evaluating on untrained data, training data is {round(training_data_weight*100)}%, {self.get_filters_description(filter_on_training, filter_on_testing)}, average on {repeat} repeats"
        )


    """
    def run_on_all (self, training_data_weight=1):
        training_indexes, remaining_indexes = partition([index for index in range(len(self.texts))], [training_data_weight])
        accuracies = []
        trained_on_texts = []
        trained_on_texts_categories = []
        for index in training_indexes:
            trained_on_texts.append(self.texts[index])
            trained_on_texts_categories.append(self.categories[index])
            self.train_detectors(trained_on_texts, trained_on_texts_categories)
            accuracies.append(self.evaluate_detectors(self.texts, self.categories))
        return accuracies

    def run_on_trained (self, training_data_weight=1):
        training_indexes, remaining_indexes = partition([index for index in range(len(self.texts))], [training_data_weight])
        accuracies = []
        trained_on_texts = []
        trained_on_texts_categories = []
        for index in training_indexes:
            trained_on_texts.append(self.texts[index])
            trained_on_texts_categories.append(self.categories[index])
            self.train_detectors(trained_on_texts, trained_on_texts_categories)
            accuracies.append(self.evaluate_detectors(trained_on_texts, trained_on_texts_categories))
        return accuracies

    def run_on_untrained (self, training_data_weight=0.8):
        training_indexes, testing_indexes = partition([index for index in range(len(self.texts))], [training_data_weight])
        accuracies = []
        testing_texts = [self.text[index] for index in testing_indexes]
        testing_texts_categories = [self.categories[index] for index in testing_indexes]
        trained_on_texts, trained_on_texts_categories = [], []
        for index in training_indexes:
            trained_on_texts.append(self.texts[index])
            trained_on_texts_categories.append(self.categories)
            self.train_detectors(trained_on_texts, trained_on_texts_categories)
            accuracies.append(self.evaluate_detectors(testing_texts, testing_texts_categories))
        return accuracies
    """