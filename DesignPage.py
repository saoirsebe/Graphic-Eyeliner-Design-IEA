import tkinter as tk
from tkinter import messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from Trial import *
from Page import Page

class DesignPage(Page):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.selected_designs = [False] * 6  # Keep track of selected designs
        self.create_widgets()

    def create_widgets(self):
        # Create the initial screen with instructions
        tk.Label(self, text="Start designing your makeup look!", font=("Arial", 18)).grid(row=0, column=0, columnspan=2, pady=20)

        # Back to Home button
        tk.Button(self, text="Back to Home", command=lambda: self.controller.show_page("HomePage")).grid(row=1, column=0, columnspan=2, pady=10)
        # Start designing button
        self.start_button = tk.Button(self, text="Start designing", command=lambda: self.start_designing())
        self.start_button.grid(row=2, column=0, columnspan=2, pady=10)

    def start_designing(self):
        self.start_button.grid_forget()
        tk.Label(self, text="Pick your favorite designs", font=("Arial", 18)).pack(pady=20)
        # Frame for the design canvas
        design_frame = tk.Frame(self)
        design_frame.grid(row=3, column=0, columnspan=2, pady=20)

        # Generate the design grid
        self.fig, self.axes = initialise_gene_pool()
        self.fig.tight_layout(pad=3)

        canvas = FigureCanvasTkAgg(self.fig, design_frame)
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=3)  # Place the canvas

        # Fill the subplots with dummy data
        self.buttons = []
        for i, ax in enumerate(self.axes.flatten()):
            ax.set_title(f"Design {i + 1}")
            button = tk.Button(
                design_frame,
                text="Select",
                bg="red",
                fg="white",
                command=lambda idx=i: self.toggle_selection(idx),
            )
            self.buttons.append(button)

            # Place each button in a grid (3 columns, 2 rows)
            row, col = divmod(i, 3)
            button.grid(row=row + 1, column=col, padx=10, pady=10, sticky="ew")

            # Add a submit button below the design buttons
        tk.Button(design_frame, text="Submit Selection", command=self.submit_selection, bg="blue", fg="white").grid(row=3, column=0, columnspan=3, pady=20)

    def toggle_selection(self, idx):
        """Toggle the selection state of a design."""
        self.selected_designs[idx] = not self.selected_designs[idx]
        # Update button appearance
        self.buttons[idx].config(
            bg="green" if self.selected_designs[idx] else "red",
            text="Selected" if self.selected_designs[idx] else "Select",
        )

    def submit_selection(self):
        """Process and validate the selected designs."""
        selected = [i for i, selected in enumerate(self.selected_designs) if selected]
        if len(selected) != 2:
            messagebox.showerror("Error", "Please select exactly 2 designs.")
        else:
            messagebox.showinfo("Selection Complete", f"You selected designs: {selected}")
            # Continue with merging or further processing