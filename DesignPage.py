import tkinter as tk
from random import randint
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from InitialiseGenePool import initialise_gene_pool
from Page import Page
import random

class DesignPage(Page):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        #self.selected_designs = [False] * 6  # Keep track of selected designs
        self.create_widgets()
        self.current_gene_pool = []
        self.selected_gene_indices = []
        self.buttons = []
        self.mutation_rate = 0.05
        self.gene_pools_all = []
        self.saved_genes_indices = []
        self.saved_genes = []
        self.saved_gene_widgets = {}
        self.current_gene_pool_figures = []

    def create_widgets(self):
        # Create the initial screen with instructions
        tk.Label(self, text="Start designing your makeup look!", font=("Arial", 18)).grid(row=0, column=2, columnspan=2, pady=20)

        # Back to Home button
        tk.Button(self, text="Back to Home", command=lambda: self.controller.show_page("HomePage")).grid(row=1, column=0, columnspan=2, pady=10)
        # Start designing button
        self.start_button = tk.Button(self, text="Start designing", command=lambda: self.start_designing())
        self.start_button.grid(row=2, column=0, columnspan=2, pady=10)

    def show_saved_genes(self):
        for widget in self.saved_gene_widgets.values():widget.destroy()
        self.saved_gene_widgets = {}
        for i, gene in enumerate(self.saved_genes):
            # Create a new frame for the gene only if it doesn't already exist
            canvas_frame = tk.Frame(self, borderwidth=1, relief="solid")
            canvas_frame.grid(row=i + 3, column=10, padx=10, pady=10, sticky="nsew")

            # Render the figure for the gene
            fig = gene.gene.render()
            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=0, column=0, columnspan=2, padx=5, pady=5)  # Place canvas at the top

            # Save the frame to the dictionary
            self.saved_gene_widgets[gene] = canvas_frame

    def re_generate(self):
        self.current_gene_pool = initialise_gene_pool()

    def start_designing(self):
        self.selected_gene_indices = []
        self.saved_genes_indices = []
        self.start_button.grid_forget()
        tk.Label(self, text="Pick your favorite designs", font=("Arial", 18)).grid(row=3, column=2, columnspan=2, pady=10)

        if not self.current_gene_pool:
            self.current_gene_pool = initialise_gene_pool()

        self.gene_pools_all.append(self.current_gene_pool) #add gene pool to list of gene pools browsed

        # Add dropdown menu for mutation rate
        tk.Label(self, text="Select Mutation Rate:", font=("Arial", 12)).grid(row=2, column=2, pady=10)
        mutation_rate_values = [round(x, 2) for x in [0.01, 0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]]
        def set_mutation_rate(value):
            self.mutation_rate = float(value)  # Update the float value
        # Create OptionMenu
        mutation_rate_dropdown = tk.OptionMenu(self, tk.StringVar(value=str(self.mutation_rate)), *mutation_rate_values,  # Dropdown options
            command=set_mutation_rate  # Update mutation rate on selection
        )
        mutation_rate_dropdown.grid(row=4, column=6, pady=10)

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

        def save_gene(index,button):
            if index in self.saved_genes_indices:
                self.saved_genes_indices.remove(index)  # Deselect the gene
                button.config(
                    highlightbackground="black",  # Default border color
                    highlightthickness=0,  # Default border thickness
                    bg="SystemButtonFace",  # Default background color
                    fg="black",  # Default text color
                    relief="raised"  # Default relief
                )
                gene_to_remove = self.current_gene_pool[index]
                self.saved_genes.remove(gene_to_remove)

                # Destroy the widget associated with this gene
                if gene_to_remove in self.saved_gene_widgets:
                    self.saved_gene_widgets[gene_to_remove].destroy()  # Destroy the frame
                    del self.saved_gene_widgets[gene_to_remove]

            else:
                self.saved_genes_indices.append(index)  # Select the gene
                button.config(
                    highlightbackground="green",  # Green border color
                    highlightthickness=2,  # Green border thickness
                    bg="lightgreen",  # Change background to light green (for example)
                    fg="black",  # Text color stays black
                    relief="sunken"  # Change the button relief to give a "pressed" feel
                )
                self.saved_genes.append(self.current_gene_pool[index])
            self.show_saved_genes()

        for i, gene in enumerate(self.current_gene_pool):
            canvas_frame = tk.Frame(self, borderwidth=1, relief="solid")
            canvas_frame.grid(row=(i // 3) + 5, column=i % 3, padx=10, pady=10, sticky="nsew")

            fig = gene.gene.render()
            self.current_gene_pool_figures.append(fig)
            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=0, column=0, columnspan=2, padx=5, pady=5)  # Place canvas at the top

            save_design_button = tk.Button(canvas_frame, text="Save Design")
            save_design_button.config(command=lambda index=i, b=save_design_button: save_gene(index, b))
            save_design_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")  # Place button below canvas

            toggle_button = tk.Button(canvas_frame, text="Toggle Selection")
            toggle_button.config(command=lambda index=i, b=toggle_button: toggle_gene(index, b))
            toggle_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")  # Place button below canvas

        re_generate_button = tk.Button(self, text="Re-generate designs", command=self.re_generate)
        re_generate_button.grid(row=10, column=0, pady=20)

        submit_button = tk.Button(self, text="Submit", command=self.submit_selection)
        submit_button.grid(row=9, column=0, pady=20)

        back_designs_button = tk.Button(self, text="Back to previous designs", command=self.back_to_prev_designs)
        back_designs_button.grid(row=10, column=0, pady=20)

        tk.Label(self, text="Saved Genes:", font=("Arial", 18)).grid(row=0, column=10, columnspan=2, pady=10)
        self.show_saved_genes()

    def submit_selection(self):
        """Submit the selected genes."""
        for fig in self.current_gene_pool_figures:
            plt.close(fig)
        selected_indices = self.selected_gene_indices
        if selected_indices:
            #selected_genes = [f"Gene {i + 1}" for i in selected_indices]
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
                    new_design = old_gene.mutate(self.mutation_rate)
                    new_gene_pool.append(new_design)
            while len(new_gene_pool) < 6:
                new_design_index = random.choice(selected_indices)
                new_design = self.current_gene_pool[new_design_index].mutate(self.mutation_rate)
                new_gene_pool.append(new_design)
            self.current_gene_pool = new_gene_pool
            self.start_designing()
        else:
            messagebox.showwarning("No Genes Selected", "No genes were selected.")

    def back_to_prev_designs(self):
        if len(self.gene_pools_all) < 2:
            messagebox.showwarning("Back Error", "Cant go back.")
        else:
            self.current_gene_pool = self.gene_pools_all[-2]
            self.gene_pools_all = self.gene_pools_all[:-1]
            self.start_designing()
