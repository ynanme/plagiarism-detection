import tkinter as tk
from tkinter import ttk

from gui.model import *
from gui.view import *
from gui.controller import *
from gui.config import *

class App (tk.Tk):

    def __init__ (self):
        super().__init__()
        self.title(TITLE)
        self.geometry(GEOMETRY)
        self.model = Model()
        self.controller = Controller(self.model)
        self.view = View(self, self.controller.get_actions())
        self.controller.set_view(self.view)
        #self.view.pack()
        self.controller.launch_app()



if __name__ == "__main__":

    app = App()
    app.mainloop()