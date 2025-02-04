"""
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

app = QApplication([])

window = QWidget()
layout = QVBoxLayout()
layout.addWidget(QLabel("Hello, PyQt!"))
window.setLayout(layout)

window.show()
app.exec()


import tkinter as tk

root = tk.Tk()
root.title("Hello Tkinter")
label = tk.Label(root, text="Hello, Tkinter!")
label.pack()
root.mainloop()


from kivy.app import App
from kivy.uix.label import Label

class MyApp(App):
    def build(self):
        return Label(text="Hello, Kivy!")

MyApp().run()
"""
"""
import webview

webview.create_window("Hello", html="<h1>Hello, PyWebview!</h1>")
webview.start()
"""

import streamlit as st

st.title("Hello, Streamlit!")
st.write("Ceci est une interface simple.")
