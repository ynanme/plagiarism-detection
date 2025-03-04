import tkinter as tk
from tkinter import ttk


class HomeScreen (ttk.Frame):

    def __init__ (self, parent):
        super().__init__(parent)
        self.init_buttons()
        self.pack_buttons()

    def init_buttons (self):
        self.buttons = [
            ttk.Button(self, text="Detector", command=self.go_to_detector),
            ttk.Button(self, text="Experimentations", command=self.go_to_experimentations),
            ttk.Button(self, text="Quit", command=self.quit)
        ]
    
    def pack_buttons (self):
        for button in self.buttons:
            button.pack()

    def go_to_detector (self):
        print("Went to detector")
    
    def go_to_experimentations (self):
        print("Went to experimentations")
    
    def quit (self):
        print("Quitted")



class DetectorScreen (ttk.Frame):

    def __init__ (self, parent):
        super().__init__(parent)
        self.config_frame = DetectorConfigFrame(self)
        self.config_frame.pack()


class DetectorConfigFrame (ttk.Frame):

    CONFIGS = [
        ("Classifier", ["Naive Bayes", "Logistic Regression", "Perceptron"], False),
        ("Featurizer", ["Count Vector", "TFIDF"], False),
        ("Tokenizer", ["Delete stopwords", "Lemmatize words"], True)
    ]

    def __init__ (self, parent):
        super().__init__()
        self.init_configs()

    def init_configs (self):
        for description, options, binary in self.CONFIGS:
            config = CheckBoxFrame(self, description, options) if binary else OptionFrame(self, description, options)
            config.pack()



class OptionFrame (ttk.Frame):

    def __init__ (self, parent, description, options):
        super().__init__()
        self.selected = tk.StringVar()
        self.init_label(description)
        self.init_menu(options)

    def init_menu (self, options):
        self.menu = ttk.OptionMenu(self, self.selected, options[0], *options, command=self.print_selected)
        self.menu.pack()

    def init_label (self, description):
        self.label = ttk.Label(self, text=description)
        self.label.pack()

    def get_selected (self):
        return self.selected.get()

    def print_selected (self, value):
        print(f"{self.get_selected()} selected")


class CheckBoxFrame (ttk.Frame):

    def __init__ (self, parent, description, options):
        super().__init__()
        self.init_label(description)
        self.init_boxes(options)

    def init_label (self, description):
        self.label = ttk.Label(self, text=description)
        self.label.pack()

    def init_boxes (self, options):
        self.boxes = [CheckBox(self, option) for option in options]
        for box in self.boxes:
            box.pack()

    def get_values (self):
        return [box.get_value() for box in self.boxes]


class CheckBox (ttk.Frame):

    def __init__ (self, parent, option):
        super().__init__(parent)
        self.variable = tk.BooleanVar()
        self.init_box(option)

    def init_box (self, option):
        self.check_button = ttk.Checkbutton(self, text=option, variable=self.variable, command=self.print_value)
        self.check_button.pack()

    def get_value (self):
        return self.variable.get()

    def print_value (self):
        print(f"box {'checked' if self.get_value() else 'unchecked'}")