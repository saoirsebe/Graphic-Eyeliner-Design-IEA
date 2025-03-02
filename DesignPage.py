from tkinter import messagebox

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from DualScrollableFrame import DualScrollableFrame
from BreedingMechanism import breed_new_designs
from InitialiseGenePool import initialise_gene_pool

class DesignPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, width=1200, height=800)
        self.number_of_rows = 3
        self.grid_propagate(False)
        self.controller = controller
        self.current_gene_pool = []
        self.selected_gene_indices = []
        self.mutation_rate = 0.1
        self.saved_genes = []
        self.saved_genes_indices = []
        self.saved_gene_widgets = {}
        self.current_gene_pool_figures = []
        self.gene_pools_previous = []
        self.create_widgets()


    def create_widgets(self):
        # Configure columns and rows.
        # Give a higher weight to rows where you want expansion.
        for i in range(6):
            self.grid_columnconfigure(i, weight=1)

        # For example, rows 4-6 (where the DualScrollableFrame sits) get extra weight.
        #self.grid_rowconfigure(0, weight=0)   # title
        #self.grid_rowconfigure(1, weight=0)   # buttons row
        #self.grid_rowconfigure(2, weight=0)   # design selection prompt
        #self.grid_rowconfigure(3, weight=0)   # mutation rate row and saved_frame starts here
        self.grid_rowconfigure(4, weight=3)   # DualScrollableFrame row (start designing)
        #self.grid_rowconfigure(5, weight=3)
        #self.grid_rowconfigure(6, weight=3)
        #self.grid_rowconfigure(7, weight=3)
        #self.grid_rowconfigure(8, weight=3)
        #self.grid_rowconfigure(9, weight=3)

        self.columnconfigure(0, weight=3)

        self.label_title = ctk.CTkLabel(self, text="Start designing your makeup look!", font=("Arial", 24))
        self.label_title.grid(row=0, column=0, columnspan=3, pady=20)

        self.back_button = ctk.CTkButton(self, text="Back to Home",
                                         command=lambda: self.controller.show_page("HomePage"))
        self.back_button.grid(row=1, column=0, pady=10, padx=10)

        self.start_button = ctk.CTkButton(self, text="Start Designing", command=self.start_designing)
        self.start_button.grid(row=1, column=1, pady=10)

        self.saved_frame = ctk.CTkScrollableFrame(self, width=250, fg_color="#2A2A2A")
        # Place saved_frame and force it to keep its size.
        self.saved_frame.grid(row=3, column=5, rowspan=5, padx=10, pady=10, sticky="nsew")
        #self.saved_frame.grid_propagate(False)
        self.saved_label = ctk.CTkLabel(self.saved_frame, text="Saved Designs", font=("Arial", 18))
        self.saved_label.pack(expand=True, fill="both")

        # This is the scrollable area for your designs.
        self.scrollable_frame = DualScrollableFrame(self, fg_color="#1A1A1A")
        self.scrollable_frame.grid(row=4, column=0, columnspan=3, rowspan=4, padx=10, pady=10, sticky="nsew")
        self.scrollable_frame.grid_propagate(False)
        self.scrollable_frame.configure(height=400)

        #Configure the inner_frame's grid so that its cells expand:
        for col in range(3):
            self.scrollable_frame.inner_frame.grid_columnconfigure(col, weight=1)

        for row in range(4, 4 + self.number_of_rows):
            self.scrollable_frame.inner_frame.grid_rowconfigure(row, weight=1)

    def finish_designing(self):
        #Add the saved designs from this page to the global list in the controller.
        self.controller.add_saved_designs(self.saved_genes)
        self.current_gene_pool =[]
        self.gene_pools_previous = []
        self.selected_gene_indices = []
        self.saved_genes = []
        self.saved_genes_indices = []
        self.saved_gene_widgets = {}
        self.current_gene_pool_figures = []
        #Then go to the SaveDesignPage.
        self.controller.show_page("SaveDesignPage")

    def add_into_pool(self, the_gene):
        if the_gene not in self.current_gene_pool:
            self.current_gene_pool.append(the_gene)
            self.start_designing()

    def un_save(self, the_gene):
        self.saved_genes.remove(the_gene)
        self.show_saved_genes()

    def show_saved_genes(self):
        for widget in self.saved_gene_widgets.values():
            widget.destroy()
        self.saved_gene_widgets = {}

        for i, gene in enumerate(self.saved_genes):
            frame = ctk.CTkFrame(self.saved_frame)
            frame.pack(pady=2.5, padx=2.5, fill='both')

            fig = gene.render_design()
            for ax in fig.get_axes():
                ax.set_axis_off()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(padx=2.5, pady=2.5)
            # Bind the canvas widget so clicking it shows the design in a popup
            #canvas_widget.bind("<Button-1>", lambda event, index=i: self.show_design_popup(index))

            into_pool_button = ctk.CTkButton(frame, text="Add back to pool", command=lambda g=gene: self.add_into_pool(g))
            into_pool_button.pack(pady=2.5)

            un_save_button = ctk.CTkButton(frame, text="Un-save", command=lambda g=gene: self.un_save(g))
            un_save_button.pack(pady=2.5)

            self.saved_gene_widgets[gene] = frame

    def re_generate(self):
        for fig, widget in self.current_gene_pool_figures:
            plt.close(fig)
            widget.destroy()
        self.gene_pools_previous.append(self.current_gene_pool)
        self.current_gene_pool = initialise_gene_pool()
        self.selected_gene_indices.clear()
        self.saved_genes_indices.clear()
        self.start_designing()

    def start_designing(self):
        self.start_button.grid_forget()
        self.current_gene_pool_figures = []

        ctk.CTkLabel(self, text="Pick your favorite designs", font=("Arial", 18)).grid(row=2, column=0, columnspan=3, pady=10)

        if not self.current_gene_pool:
            self.current_gene_pool = initialise_gene_pool()

        if len(self.current_gene_pool) ==6:
            self.number_of_rows = 3
        else:
            self.number_of_rows = 4

        # Mutation rate selection
        ctk.CTkLabel(self, text="Select Mutation Rate:", font=("Arial", 12)).grid(row=3, column=0, pady=10)
        mutation_rate_values = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11,0.12,0.13,0.14,0.15]
        mutation_rate_dropdown = ctk.CTkOptionMenu(self, values=[str(x) for x in mutation_rate_values],command=lambda val: self.set_mutation_rate(val))
        mutation_rate_dropdown.set(str(self.mutation_rate))
        mutation_rate_dropdown.grid(row=3, column=1, pady=10)

        for i, gene in enumerate(self.current_gene_pool):
            frame = ctk.CTkFrame(self.scrollable_frame.inner_frame)
            frame.grid(row=(i // 3) + 5, column=i % 3, padx=10, pady=10, sticky="nsew")

            fig = gene.render_design()
            for ax in fig.get_axes():
                ax.set_axis_off()

            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()

            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(expand=True, fill="both", padx=2.5, pady=2.5)
            self.current_gene_pool_figures.append((fig, frame))

            #Bind the canvas widget so clicking it shows the design in a popup
            canvas_widget.bind("<Button-1>", lambda event, index=i: self.show_design_popup(index))

            toggle_button = ctk.CTkButton(frame, text="Toggle Selection")
            toggle_button.configure(command=lambda index=i, b=toggle_button: self.toggle_gene(index, b))
            toggle_button.pack(pady=2.5)

            save_button = ctk.CTkButton(frame, text="Save")
            save_button.configure(command=lambda index=i, b=save_button: self.save_gene(index, b))
            save_button.pack(pady=2.5)


        ctk.CTkButton(self, text="Re-generate designs", command=self.re_generate).grid(row=11, column=1, pady=5)
        ctk.CTkButton(self, text="Submit", command=self.submit_selection).grid(row=10, column=0, pady=5)
        ctk.CTkButton(self, text="Back to previous designs", command=self.back_to_prev_designs).grid(row=11, column=0, pady=5)
        ctk.CTkButton(self, text="Finish Designing", command=self.finish_designing) \
            .grid(row=12, column=0, columnspan=3, pady=20, sticky="ew")
        self.show_saved_genes()

    def set_mutation_rate(self, value):
        self.mutation_rate = float(value)

    def toggle_gene(self, index, button):
        if index in self.selected_gene_indices:
            self.selected_gene_indices.remove(index)
            button.configure(
                text="Toggle Selection",
                fg_color="gray",  # Default background
                text_color="black"  # Default text color
            )
        else:
            self.selected_gene_indices.append(index)
            button.configure(
                text="Selected",  # Change the button label
                fg_color="lightgreen",  # Background to indicate selection
                text_color="black"  # Text stays black
            )

    def save_gene(self, index, button):
        gene = self.current_gene_pool[index]
        if gene in self.saved_genes:
            self.saved_genes.remove(gene)
            # Destroy the widget associated with this gene
            if gene in self.saved_gene_widgets:
                self.saved_gene_widgets[gene].destroy()  # Destroy the frame
                del self.saved_gene_widgets[gene]
            button.configure(
                text="Save",
                fg_color="gray",  # Default background
                text_color="black"  # Default text color
            )
        else:
            self.saved_genes.append(gene)
            button.configure(
                text="Saved",  # Change the button label
                fg_color="lightgreen",  # Background to indicate selection
                text_color="black"  # Text stays black
            )
        self.show_saved_genes()

    def submit_selection(self):
        for fig , widget in self.current_gene_pool_figures:
            plt.close(fig)
            widget.destroy()

        if self.selected_gene_indices:
            self.gene_pools_previous.append(self.current_gene_pool)
            selected_genes = [self.current_gene_pool[i] for i in self.selected_gene_indices]
            self.current_gene_pool = breed_new_designs(selected_genes, self.mutation_rate)
            self.selected_gene_indices.clear()
            self.saved_genes_indices.clear()
            self.start_designing()
        else:
            messagebox.showwarning("No Genes Selected", "No genes were selected.")

    def back_to_prev_designs(self):
        if len(self.gene_pools_previous) < 1:
            messagebox.showwarning("Back Error", "Cannot go back.")
        else:
            self.current_gene_pool = self.gene_pools_previous[-1]
            self.gene_pools_previous = self.gene_pools_previous[:-1]
            self.start_designing()

    def show_design_popup(self, index):
        # Create a popup window
        popup = ctk.CTkToplevel(self)
        popup.title("Design Preview")
        popup.geometry("600x400")
        #Bind event to close the popup when clicking outside
        popup.bind("<FocusOut>", lambda event: popup.destroy())

        # Create a frame for the content
        content_frame = ctk.CTkFrame(popup)
        content_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Render a larger version of the design. You might want to adjust gene.render_design()
        # to account for the new size if necessary.
        large_fig = self.current_gene_pool[index].render_design(scale = 2.5)
        for ax in large_fig.get_axes():
            ax.set_axis_off()
        large_canvas = FigureCanvasTkAgg(large_fig, master=content_frame)
        large_canvas.draw()
        large_canvas.get_tk_widget().pack(expand=True, fill="both", padx=5, pady=5)

        # Create a close button to dismiss the popup
        close_button = ctk.CTkButton(content_frame, text="Close", command=popup.destroy)
        close_button.pack(pady=5)
