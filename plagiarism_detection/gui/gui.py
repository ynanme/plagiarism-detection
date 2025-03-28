import tkinter as tk
from tkinter import ttk

class CarouselFrame(ttk.Frame):
    def __init__(self, parent, frames, titles=None):
        super().__init__(parent)

        self.frames = frames
        self.titles = titles if titles else [f"Frame {i+1}" for i in range(len(frames))]
        self.current_index = 0

        # Header avec Previous, Titre et Next
        self.header = ttk.Frame(self)
        self.header.pack(fill="x", padx=10, pady=5)

        self.btn_previous = ttk.Button(self.header, text="⬅ Previous", command=self.show_previous)
        self.btn_previous.pack(side="left")

        self.title_label = ttk.Label(self.header, text=self.titles[self.current_index], anchor="center")
        self.title_label.pack(side="left", expand=True)

        self.btn_next = ttk.Button(self.header, text="Next ➡", command=self.show_next)
        self.btn_next.pack(side="right")

        # Conteneur pour afficher les frames
        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True)

        # Afficher la première frame
        self.frames[self.current_index].pack(fill="both", expand=True)
        self.update_buttons()

    def show_previous(self):
        """Affiche la frame précédente"""
        if self.current_index > 0:
            self.frames[self.current_index].pack_forget()
            self.current_index -= 1
            self.frames[self.current_index].pack(fill="both", expand=True)
            self.title_label.config(text=self.titles[self.current_index])
            self.update_buttons()

    def show_next(self):
        """Affiche la frame suivante"""
        if self.current_index < len(self.frames) - 1:
            self.frames[self.current_index].pack_forget()
            self.current_index += 1
            self.frames[self.current_index].pack(fill="both", expand=True)
            self.title_label.config(text=self.titles[self.current_index])
            self.update_buttons()

    def update_buttons(self):
        """Active/Désactive les boutons en fonction de la position"""
        self.btn_previous.config(state="normal" if self.current_index > 0 else "disabled")
        self.btn_next.config(state="normal" if self.current_index < len(self.frames) - 1 else "disabled")

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("CarouselFrame Demo")
    root.geometry("400x300")

    # Création de plusieurs frames de test
    frames = [
        ttk.Frame(root, width=400, height=250),
        ttk.Frame(root, width=400, height=250),
        ttk.Frame(root, width=400, height=250)
    ]
    
    # Ajout de labels pour identifier les frames
    for i, frame in enumerate(frames):
        label = ttk.Label(frame, text=f"Contenu de la Frame {i+1}", font=("Arial", 14))
        label.pack(expand=True)

    carousel = CarouselFrame(root, frames, ["Page 1", "Page 2", "Page 3"])
    carousel.pack(fill="both", expand=True)

    root.mainloop()
