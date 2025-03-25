from experimentations.experiments import load_data
from gui.config import *


class Controller ():

    def __init__ (self, model, view=None):
        self._model = model
        self._view = view
        self.gather_actions()

    def set_view (self, view):
        self._view = view

    def control (self):
        self._model.gets_fed(load_data(DATA_DIRECTORY))
        self._view.show_home()


    def gather_actions (self):
        self._actions = {
            "home": [self.go_to_detector, self.go_to_experimentations, self.quit],
            "detector": [self.update_param, self.add_detector],
            "experiments": [self.update_param, self.add_detector_to_experiments]
        }
    
    def get_actions (self):
        return self._actions

    def go_to_detector (self):
        self._view.show_detector()
    
    def go_to_experimentations (self):
        self._view.show_experiments()
    
    def quit (self):
        print("Quitted")

    def update_param (self, id_, value):
        #id_, value = param
        self._model.detector_is_being_built(id_, value)

    def add_detector (self):
        self._model.add_detector()

    def add_detector_to_experiments (self):
        self._model.add_detector_to_experiments()
        return self._model.get_last_experiment()