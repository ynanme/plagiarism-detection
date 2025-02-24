import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Pack, Grid et Place")

frame = ttk.Frame(root, padding=10)
frame.pack(fill="both", expand=True)

# Boutons avec `pack`
btn1 = ttk.Button(frame, text="Pack")
btn1.pack(side="top", fill="x", pady=5)

# Grille
grid_frame = ttk.Frame(frame)
grid_frame.pack(pady=10)
btn2 = ttk.Button(grid_frame, text="Grid 1")
btn2.grid(row=0, column=0, padx=5)
btn3 = ttk.Button(grid_frame, text="Grid 2")
btn3.grid(row=0, column=1, padx=5)

# Bouton avec `place`
btn4 = ttk.Button(frame, text="Place")
btn4.place(x=50, y=100)

root.mainloop()
