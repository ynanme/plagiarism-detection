from experimentations.experiments_utils import load_data
from gui.config import *


class Controller ():

    def __init__ (self, model, view=None):
        self._model = model
        self._view = view
        self.gather_actions()

    def set_view (self, view):
        self._view = view

    def launch_app (self):
        self._model.gets_fed(load_data(DATA_DIRECTORY))
        self._view.show_home()


    def gather_actions (self):
        self._actions = {
            "home": [self.go_to_experimentations],
            "experiments": [self.update_param, self.add_detector, self.add_detectors_to_experiments, self.clear_experiments]
        }
    
    def get_actions (self):
        return self._actions
    
    def go_to_experimentations (self):
        self._view.show_experiments()


    def update_param (self, id_, value):
        #id_, value = param
        self._model.detector_is_being_built(id_, value)

    def add_detector (self):
        self._model.add_detector()
        return self._model.get_last_experiment()

    def add_detectors_to_experiments (self):
        self._model.add_detectors_to_experiments()
        return self._model.get_last_experiment()

    def clear_experiments (self):
        self._model.clear()