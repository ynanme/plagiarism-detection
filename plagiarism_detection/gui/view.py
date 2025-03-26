import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.config import *


class View (ttk.Frame):

    def __init__ (self, parent, actions):
        super().__init__(parent)
        self.home_screen = HomeScreen(self, actions["home"])
        self.detector_screen = DetectorScreen(self, actions["detector"])
        self.experiments_screen = ExperimentsScreen(self, actions["experiments"])
        self.pack()

    def show_home (self):
        self.detector_screen.pack_forget()
        self.experiments_screen.pack_forget()
        self.home_screen.pack()

    def show_detector (self):
        self.home_screen.pack_forget()
        self.experiments_screen.pack_forget()
        self.detector_screen.pack()

    def show_experiments (self):
        self.home_screen.pack_forget()
        self.detector_screen.pack_forget()
        self.experiments_screen.pack()



class GridFrame(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)

    def pack_grid (self, mode, composants, repartitions):

        if mode not in ["row", "column"]:
            raise ValueError("Le mode doit être 'row' ou 'column'.")

        if not repartitions:
            raise ValueError("Les répartitions ne peuvent pas être vides.")

        num_cols = len(repartitions)  # Nombre de colonnes par ligne

        for index, comp in enumerate(composants):
            row, col = divmod(index, num_cols)
            comp.grid(row=row, column=col, sticky="nsew")

        for col, weight in enumerate(repartitions):
            self.grid_columnconfigure(col, weight=weight)

        for row in range((len(composants) + num_cols - 1) // num_cols):
            self.grid_rowconfigure(row, weight=1)



class HomeScreen (ttk.Frame):

    def __init__ (self, parent, actions):
        super().__init__(parent)
        self.init_buttons(actions)
        self.pack_buttons()

    def init_buttons (self, actions):
        self.buttons = [
            ttk.Button(self, text="Detector", command=actions[0]),
            ttk.Button(self, text="Experimentations", command=actions[1]),
            ttk.Button(self, text="Quit", command=actions[2])
        ]
    
    def pack_buttons (self):
        for button in self.buttons:
            button.pack()



class DetectorScreen (ttk.Frame):

    def __init__ (self, parent, config_actions):
        super().__init__(parent)
        self.config_frame = DetectorConfigFrame(self, config_actions)
        self.config_frame.pack()


class ExperimentsScreen (GridFrame):

    def __init__ (self, parent, detector_config_actions):
        super().__init__(parent)
        self.detector_config_frame = DetectorConfigFrame(self, detector_config_actions)
        self.experiments_frame = ExperimentsFrame(self, detector_config_actions)
        self.pack_grid("column", [self.detector_config_frame, self.experiments_frame], [2, 8])



class DetectorConfigFrame (ttk.Frame):

    def __init__ (self, parent, actions):
        super().__init__(parent)
        self.init_configs(actions)
        self.adding_detector_button = ttk.Button(self, text="Ajouter", command=actions[1])
        self.adding_detector_button.pack()

    def init_configs (self, actions):
        for config, options in DETECTOR_M_ARY_CONFIGS:
            config_frame = OptionFrame(self, config, [option[1] for option in options], actions[0])
            config_frame.pack()
        for config, options in DETECTOR_BINARY_CONFIGS:
            config_frame = CheckBoxFrame(self, config, options, actions[0])
            config_frame.pack()
        for config in DETECTOR_INPUT_CONFIGS:
            config_frame = InputFrame(self, config, actions[0])
            config_frame.pack()


class ExperimentsFrame (ttk.Frame):

    def __init__ (self, parent, actions):
        super().__init__(parent)
        self.plots_frame = PlotFrame(self)
        self.launch_experiment = actions[1]
        self.plots_frame.pack()
        self.adding_detector_button = ttk.Button(self, text="Ajouter", command=self.update_results)
        self.adding_detector_button.pack()
        self.clearing_button = ttk.Button(self, text="Clear", command=self.clear)
        self.clearing_button.pack()
        self.init_infos_frame()

    def init_infos_frame (self):
        self.detectors_infos_frame = ListFrame(self, "CURRENT DETECTORS")
        self.detectors_infos_frame.pack()

    def update_results (self):
        experiment = self.launch_experiment()
        self.draw_plots(experiment["results"])
        self.add_infos(experiment["infos"])

    def draw_plots (self, results):
        for X, Y in results:
            self.plots_frame.add_curve(X, Y)

    def add_infos (self, infos):
        for name, detector in infos:
            self.detectors_infos_frame.add_item(name, detector)

    def clear (self):
        self.detectors_infos_frame.pack_forget()
        self.init_infos_frame()
        self.plots_frame.remove_curves()


class Param ():

    def __init__ (self, ref, action):
        self.ref = ref
        self.action = action

    def gets_updated (self, value=None):
        return self.action(self.ref, value)


class OptionFrame (ttk.Frame, Param):

    def __init__ (self, parent, refs, options, action):
        ttk.Frame.__init__(self, parent)
        Param.__init__(self, refs[0], action)
        self.selected = tk.StringVar()
        self.init_label(refs[1])
        self.init_menu(options)

    def init_menu (self, options):
        self.menu = ttk.OptionMenu(self, self.selected, options[0], *options, command=self.gets_updated)
        self.menu.pack()

    def init_label (self, description):
        self.label = ttk.Label(self, text=description)
        self.label.pack()

    def get_selected (self):
        return self.selected.get()

    def print_selected (self, value):
        print(f"{self.get_selected()} selected")


class CheckBoxFrame (ttk.Frame):

    def __init__ (self, parent, description, options, action):
        super().__init__(parent)
        self.init_label(description)
        self.init_boxes(options, action)

    def init_label (self, description):
        self.label = ttk.Label(self, text=description)
        self.label.pack()

    def init_boxes (self, options, action):
        self.boxes = [CheckBox(self, option, action) for option in options]
        for box in self.boxes:
            box.pack()

    def get_values (self):
        return [box.get_value() for box in self.boxes]


class CheckBox (ttk.Frame, Param):

    def __init__ (self, parent, refs, action):
        ttk.Frame.__init__(self, parent)
        Param.__init__(self, refs[0], action)
        self.action = action
        self.variable = tk.BooleanVar()
        self.init_box(refs[1])

    def init_box (self, option):
        self.check_button = ttk.Checkbutton(self, text=option, variable=self.variable, command=self.report_update)
        self.check_button.pack()

    def get_value (self):
        return self.variable.get()

    def report_update (self):
        self.gets_updated(self.get_value())

    def print_value (self):
        print(f"box {'checked' if self.get_value() else 'unchecked'}")



class InputFrame (ttk.Frame, Param):

    def __init__ (self, parent, refs, action):
        ttk.Frame.__init__(self, parent)
        Param.__init__(self, refs[0], action)
        self.init_label(refs[1])
        self.init_input()

    def init_label (self, description):
        self.label = ttk.Label(self, text=description)
        self.label.pack()

    def init_input (self):
        self.variable = tk.StringVar()
        self.variable.trace_add("write", self.report_change)
        self.entry = ttk.Entry(self, textvariable=self.variable)
        self.entry.pack()

    def report_change (self, *args):
        self.gets_updated(self.variable.get())



class PlotFrame (ttk.Frame):

    def __init__ (self, parent):
        super().__init__(parent)
        self.fig, self.ax = plt.subplots(figsize=PLOTS_FIG_SIZE)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.curves = []
        self.canvas.get_tk_widget().pack()

    def add_curve (self, X, Y):
        #curves = self.ax.plot(X, Y)
        self.curves = self.ax.plot(X, Y)
        self.canvas.draw()

    def add_curves (self, X, Ys):
        for Y in Ys:
            self.add_curve(X, Y)

    def remove_curves (self):
        for curve in self.curves:
            curve.remove()
            self.ax.legend()
            self.canvas.draw()
        self.curves = []


class ListFrame (ttk.Frame):

    def __init__ (self, parent, title, items=[]):
        super().__init__(parent)
        self.items_count = tk.StringVar()
        self.items_count_label = ttk.Label(self, textvariable=self.items_count, text=0)
        self.items_count_label.pack()
        self.title_label = ttk.Label(self, text=title)
        self.title_label.pack()
        self.delete_all_button = ttk.Button(self, text="Delete All", command=self.delete_all)
        self.init_items_frames(items)

    def init_items_frames (self, items):
        self.items_frames = []
        for item in items:
            self.add_item(item)

    def add_item (self, id_, item):
        item_frame = ItemFrame(self, id_, item)
        self.items_frames.append(item_frame)
        self.items_count.set(len(self.items_frames))
        item_frame.pack()

    def delete_all (self):
        for item_frame in self.items_frames:
            item_frame.pack_forget()
        self.items_frames = []

class ItemFrame (ttk.Frame):

    def __init__ (self, parent, id_, description):
        super().__init__(parent)
        self._id = id_
        self.id_label = ttk.Label(self, text=self._id)
        self.description_label = ttk.Label(self, text=description)
        self.id_label.pack()
        self.description_label.pack()


