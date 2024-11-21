import tkinter as tk
from tkinter import messagebox
from DesignPage import DesignPage
from HomePage import HomePage
from LoginPage import LoginPage
from LoginPage import SignUpPage
from Page import Page
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.constants import *

class App(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")

        self.title("Eyeliner Design App")
        self.geometry("1350x600")

        # Root Canvas and Scrollbars
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Set canvas size and scrolling region
        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.canvas = tk.Canvas(self, scrollregion=f"0 0 {w * 2} {h * 2}")
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)

        # Add vertical and horizontal scrollbars
        self.makescroll(self, self.canvas)
        self.bind_mousewheel()

        # Create a container frame for all pages
        self.container = tk.Frame(self.canvas)
        self.container.bind("<Configure>", self.update_scrollregion)  # Dynamically update scroll region
        self.canvas.create_window((0, 0), window=self.container, anchor="nw")

        self.page_classes = {
            "HomePage": HomePage,
            "DesignPage": DesignPage,
            "LoginPage": LoginPage,
            "SignUpPage": SignUpPage,
        }

        self.pages = {}  # Dictionary to store pages
        for PageClass in (HomePage, DesignPage, SignUpPage, LoginPage):
            page_name = PageClass.__name__
            page = PageClass(parent=self.container, controller=self)  # Use page_frame instead of container
            self.pages[page_name] = page
            page.grid(row=0, column=0, sticky="nsew")

        # Initialize pages
        self.show_page("DesignPage")

    def makescroll(self, parent, thing):
        """Adds scrollbars to the given canvas"""
        v = tk.Scrollbar(parent, orient=tk.VERTICAL, command=thing.yview)
        v.grid(row=0, column=1, sticky=tk.NS)
        thing.config(yscrollcommand=v.set)
        h = tk.Scrollbar(parent, orient=tk.HORIZONTAL, command=thing.xview)
        h.grid(row=1, column=0, sticky=tk.EW)
        thing.config(xscrollcommand=h.set)

    def update_scrollregion(self, event):
        """Update canvas scroll region based on the container's size"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def bind_mousewheel(self):
        """Enable mousewheel scrolling for the canvas."""
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)

    def on_mousewheel(self, event):
        """Scroll the canvas with the mousewheel."""
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def show_page(self, page_name):
        """Show the given page"""
        if page_name not in self.pages:
            page_class = self.page_classes[page_name]
            page = page_class(parent=self.container, controller=self)
            self.pages[page_name] = page
            page.grid(row=0, column=0, sticky="nsew")

        # Hide all pages and show the selected one
        for widget in self.container.winfo_children():
            widget.grid_forget()
        self.pages[page_name].grid(row=0, column=0, sticky="nsew")

App().mainloop()
