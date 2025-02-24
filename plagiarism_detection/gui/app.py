import tkinter as tk
from tkinter import ttk

from view import HomeScreen, DetectorScreen


class App (tk.Tk):

    TITLE = "PLAGIARISM DETECTION"
    GEOMETRY = "800x600"

    def __init__ (self):
        super().__init__()
        self.title(self.TITLE)
        self.geometry(self.GEOMETRY)
        self.home_screen = DetectorScreen(self)
        self.home_screen.pack()



if __name__ == "__main__":

    app = App()
    app.mainloop()