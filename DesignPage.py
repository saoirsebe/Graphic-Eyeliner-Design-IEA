import tkinter as tk
from random import randint
from tkinter import messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from Trial import *
from Page import Page

class DesignPage(Page):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        #self.selected_designs = [False] * 6  # Keep track of selected designs
        self.create_widgets()
        self.current_gene_pool = []
        self.selected_gene_indices = []
        self.buttons = []

    def create_widgets(self):
        # Create the initial screen with instructions
        tk.Label(self, text="Start designing your makeup look!", font=("Arial", 18)).grid(row=0, column=2, columnspan=2, pady=20)

        # Back to Home button
        tk.Button(self, text="Back to Home", command=lambda: self.controller.show_page("HomePage")).grid(row=1, column=0, columnspan=2, pady=10)
        # Start designing button
        self.start_button = tk.Button(self, text="Start designing", command=lambda: self.start_designing())
        self.start_button.grid(row=2, column=0, columnspan=2, pady=10)

    def start_designing(self):
        self.selected_gene_indices = []
        self.start_button.grid_forget()
        tk.Label(self, text="Pick your favorite designs", font=("Arial", 18)).grid(row=3, column=2, columnspan=2, pady=10)
        if not self.current_gene_pool:
            self.current_gene_pool = initialise_gene_pool()

        def toggle_gene(index,button):
            """Toggle the selection of a gene."""
            if index in self.selected_gene_indices:
                self.selected_gene_indices.remove(index)  # Deselect the gene
                button.config(
                    highlightbackground="black",  # Default border color
                    highlightthickness=0,  # Default border thickness
                    bg="SystemButtonFace",  # Default background color
                    fg="black",  # Default text color
                    relief="raised"  # Default relief
                )
            else:
                self.selected_gene_indices.append(index)  # Select the gene
                button.config(
                    highlightbackground="green",  # Green border color
                    highlightthickness=2,  # Green border thickness
                    bg="lightgreen",  # Change background to light green (for example)
                    fg="black",  # Text color stays black
                    relief="sunken"  # Change the button relief to give a "pressed" feel
                )

        for i, gene in enumerate(self.current_gene_pool):
            canvas_frame = tk.Frame(self, borderwidth=1, relief="solid")
            canvas_frame.grid(row=(i // 3) + 5, column=i % 3, padx=10, pady=10, sticky="nsew")

            fig = gene.render()
            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=0, column=0, columnspan=2, padx=5, pady=5)  # Place canvas at the top

            button = tk.Button(canvas_frame, text="Toggle Selection")
            button.config(command=lambda index=i, b=button: toggle_gene(index, b))
            button.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")  # Place button below canvas

        submit_button = tk.Button(self, text="Submit", command=self.submit_selection)
        submit_button.grid(row=9, column=0, pady=20)


    def submit_selection(self):
        """Submit the selected genes."""
        selected_indices = self.selected_gene_indices
        print(self.selected_gene_indices)
        if selected_indices:
            selected_genes = [f"Gene {i + 1}" for i in selected_indices]
            messagebox.showinfo("Selected Genes", f"Genes selected: {', '.join(selected_genes)}")
            n_selected = len(selected_indices)
            if n_selected == 1 or 2 or 3 or 6:
                n_of_each = 6 // len(selected_indices)
            elif n_selected == 4 or 5:
                n_of_each = 1
            else:
                print("Whatt")
            new_gene_pool = []
            for selected in selected_indices:
                old_gene = self.current_gene_pool[selected]
                for i in range(n_of_each):
                    new_design = old_gene.mutate()
                    new_gene_pool.append(new_design)
            while len(new_gene_pool) < 6:
                new_design = self.current_gene_pool[random.randint(0,5)].mutate()
                new_gene_pool.append(new_design)
            self.current_gene_pool = new_gene_pool
            self.start_designing()
        else:
            messagebox.showwarning("No Genes Selected", "No genes were selected.")


