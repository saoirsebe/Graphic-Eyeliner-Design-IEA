import tkinter as tk
import ttkbootstrap as tb
from ttkbootstrap.constants import *

# Base class for all pages
class Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller


    def show(self):
        self.grid(row=0, column=0, sticky="nsew")  # Use grid instead of pack
