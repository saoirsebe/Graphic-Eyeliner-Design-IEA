import asyncio
from tkinter import messagebox
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from DualScrollableFrame import DualScrollableFrame
from BreedingMechanism import breed_new_designs, breed_new_designs_with_auto_selection
from InitialiseGenePool import initialise_gene_pool

class DesignPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        # Use a light background for a modern look.
        super().__init__(parent, width=1250, height=800, fg_color="#F9FAFB")
        self.number_of_rows = 3
        self.grid_propagate(False)
        self.controller = controller
        self.current_gene_pool = []
        self.selected_gene_indices = []
        self.mutation_rate = 0.06
        self.saved_genes = []
        self.saved_genes_indices = []
        # saved_gene_widgets now stores tuples: (fig, widget)
        self.saved_gene_widgets = {}
        self.current_gene_pool_figures = []
        self.gene_pools_previous = []
        self.create_widgets()

    def create_widgets(self):
        # Configure columns (6 columns for balanced layout)
        for i in range(6):
            self.grid_columnconfigure(i, weight=1)
        self.columnconfigure(0, weight=3)
        self.grid_rowconfigure(0, weight=0)  # header
        self.grid_rowconfigure(1, weight=0)  # navigation buttons
        self.grid_rowconfigure(2, weight=0)  # instruction area (added later)
        self.grid_rowconfigure(3, weight=0)  # mutation rate row
        self.grid_rowconfigure(4, weight=3)  # main gene pool area
        self.grid_rowconfigure(5, weight=0)

        # ---------- Header Section ----------
        # Title Label
        self.label_title = ctk.CTkLabel(
            self, text="Start designing your makeup look!",
            font=("Helvetica", 28, "bold"),
            text_color="#111111"
        )
        self.label_title.grid(row=0, column=0, columnspan=3, pady=20)

        # Navigation Buttons
        self.back_button = ctk.CTkButton(
            self, text="Back to Home",
            command=lambda: self.go_home(),
            font=("Helvetica", 14),
            fg_color="#3B8ED0",
            hover_color="#1C6EA4"
        )
        self.back_button.grid(row=1, column=0, pady=10, padx=10)

        self.start_button = ctk.CTkButton(
            self, text="Start Designing",
            command=self.start_designing,
            font=("Helvetica", 14),
            fg_color="#111111",
            hover_color="#333333",
            text_color="#FFFFFF",
            corner_radius=8
        )
        self.start_button.grid(row=1, column=1, pady=10)

        # ---------- Saved Designs Side Panel ----------
        # Use a rounded colored frame on the right.
        self.saved_frame = ctk.CTkScrollableFrame(
            self, width=250, fg_color="#FBCFD2", corner_radius=20
        )
        # Position saved_frame to occupy columns 5 (right side), spanning rows 3-7.
        self.saved_frame.grid(row=3, column=5, rowspan=5, padx=10, pady=10, sticky="nsew")
        self.saved_label = ctk.CTkLabel(
            self.saved_frame, text="Saved Designs",
            font=("Helvetica", 18, "bold"),
            text_color="#111111"
        )
        self.saved_label.pack(expand=True, fill="x", pady=(10,5))
        # The inner saved designs will be shown via self.show_saved_genes() below.

        # ---------- Main Design Pool (Scrollable Area) ----------
        # Outer frame for styling
        self.outer_scroll_frame = ctk.CTkFrame(self, height=520, fg_color="#E8E8E8", corner_radius=15)
        self.outer_scroll_frame.grid(row=4, column=0, columnspan=3, rowspan=4, padx=10, pady=10, sticky="nsew")
        self.outer_scroll_frame.grid_propagate(False)
        # DualScrollableFrame inside with a white background
        self.scrollable_frame = DualScrollableFrame(self, fg_color="#FFFFFF")
        self.scrollable_frame.grid(row=4, column=0, columnspan=3, rowspan=4, padx=5, pady=5, sticky="nsew")
        self.scrollable_frame.grid_propagate(False)
        self.scrollable_frame.configure(height=500)
        self.rowconfigure(4, minsize=500)

        # Configure the inner_frame's grid so that its cells expand:
        for col in range(3):
            self.scrollable_frame.inner_frame.grid_columnconfigure(col, weight=1)

        for row in range(4, 4 + self.number_of_rows):
            self.scrollable_frame.inner_frame.grid_rowconfigure(row, weight=1)

    def go_home(self):
        # Close and destroy current gene pool figures
        for fig, widget in self.current_gene_pool_figures:
            plt.close(fig)
            widget.destroy()
        self.current_gene_pool_figures = []
        # Close and destroy saved design figures
        for fig, widget in self.saved_gene_widgets.values():
            plt.close(fig)
            widget.destroy()
        self.saved_gene_widgets = {}
        # Reset design lists and indices
        self.current_gene_pool = []
        self.gene_pools_previous = []
        self.selected_gene_indices = []
        self.saved_genes = []
        self.saved_genes_indices = []
        self.controller.pages["HomePage"].show_recent_designs()
        self.controller.show_page("HomePage")

    def finish_designing(self):
        # Close and destroy current gene pool figures
        for fig, widget in self.current_gene_pool_figures:
            plt.close(fig)
            widget.destroy()
        self.current_gene_pool_figures = []
        # Close and destroy saved design figures
        for fig, widget in self.saved_gene_widgets.values():
            plt.close(fig)
            widget.destroy()
        self.saved_gene_widgets = {}
        # Add the saved designs from this page to the global list in the controller.
        self.controller.add_saved_designs(self.saved_genes)
        if self.controller.current_user:
            self.controller.save_user_designs(self.controller.current_user,
                                              self.controller.all_saved_designs)
        # Reset lists and indices
        self.current_gene_pool = []
        self.gene_pools_previous = []
        self.selected_gene_indices = []
        self.saved_genes = []
        self.saved_genes_indices = []
        self.controller.show_page("SaveDesignPage")

    def add_into_pool(self, the_gene):
        if the_gene not in self.current_gene_pool:
            self.current_gene_pool.append(the_gene)
            self.start_designing()

    def un_save(self, the_gene):
        if the_gene in self.saved_genes:
            self.saved_genes.remove(the_gene)
        self.show_saved_genes()

    def show_saved_genes(self):
        # Close and destroy any existing saved design figures/widgets
        for fig, widget in self.saved_gene_widgets.values():
            plt.close(fig)
            widget.destroy()
        self.saved_gene_widgets = {}

        for i, gene in enumerate(self.saved_genes):
            frame = ctk.CTkFrame(self.saved_frame, fg_color="#FFFFFF", corner_radius=10)
            frame.pack(pady=2.5, padx=2.5, fill='both')
            fig = gene.render_design()
            for ax in fig.get_axes():
                ax.set_axis_off()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(padx=5, pady=5)
            # Buttons to add back to pool or un-save.
            into_pool_button = ctk.CTkButton(
                frame, text="Add back to pool",
                command=lambda g=gene: self.add_into_pool(g),
                font=("Helvetica", 10),
                fg_color="#3B8ED0",
                hover_color="#1C6EA4"
            )
            into_pool_button.pack(pady=2)
            un_save_button = ctk.CTkButton(
                frame, text="Un-save",
                command=lambda g=gene: self.un_save(g),
                font=("Helvetica", 10),
                fg_color="#D9534F",
                hover_color="#C9302C"
            )
            un_save_button.pack(pady=2)
            # Store both the figure and its widget for later cleanup
            self.saved_gene_widgets[gene] = (fig, frame)

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

        ctk.CTkLabel(self, text="Pick your favorite designs", font=("Helvetica", 18), text_color="#111111").grid(row=2, column=0, columnspan=3, pady=10)

        if not self.current_gene_pool:
            self.current_gene_pool = initialise_gene_pool()

        if len(self.current_gene_pool) == 6:
            self.number_of_rows = 3
        else:
            self.number_of_rows = 4

        # Mutation rate selection
        ctk.CTkLabel(self, text="Select Mutation Rate:", font=("Arial", 12)).grid(row=3, column=0, pady=10)
        mutation_rate_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
        mutation_rate_dropdown = ctk.CTkOptionMenu(self, values=[str(x) for x in mutation_rate_values],
                                                   command=lambda val: self.set_mutation_rate(val))
        mutation_rate_dropdown.set(str(self.mutation_rate))
        mutation_rate_dropdown.grid(row=3, column=1, pady=10)

        # Display designs in the scrollable gene pool area
        for i, gene in enumerate(self.current_gene_pool):
            frame = ctk.CTkFrame(self.scrollable_frame.inner_frame, fg_color="#FFFFFF", corner_radius=10)
            frame.grid(row=(i // 3) + 5, column=i % 3, padx=2, pady=2, sticky="nsew")

            fig = gene.render_design()
            for ax in fig.get_axes():
                ax.set_axis_off()

            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(expand=True, fill="both", padx=2, pady=2)
            self.current_gene_pool_figures.append((fig, frame))

            # Bind click event to show a larger preview
            canvas_widget.bind("<Button-1>", lambda event, index=i: self.show_design_popup(index))

            toggle_button = ctk.CTkButton(
                frame, text="Select design", font=("Helvetica", 11), width=90, height=25,
                fg_color="gray", hover_color="#888888"
            )
            toggle_button.configure(command=lambda index=i, b=toggle_button: self.toggle_gene(index, b))
            toggle_button.pack(pady=2)

            save_button = ctk.CTkButton(
                frame, text="Save", font=("Helvetica", 11), width=90, height=25,
                fg_color="gray", hover_color="#888888"
            )
            save_button.configure(command=lambda index=i, b=save_button: self.save_gene(index, b))
            save_button.pack(pady=2)

        # ---------- Bottom Buttons ----------
        self.regenerate_button = ctk.CTkButton(
            self, text="Re-generate designs",
            command=self.re_generate,
            font=("Helvetica", 14),
            fg_color="#3B8ED0",
            hover_color="#1C6EA4"
        )
        self.regenerate_button.grid(row=11, column=1, pady=5)

        self.submit_button = ctk.CTkButton(
            self, text="Submit",
            command=self.submit_selection,
            font=("Helvetica", 14),
            fg_color="#3B8ED0",
            hover_color="#1C6EA4"
        )
        self.submit_button.grid(row=10, column=0, pady=5)

        self.back_prev_button = ctk.CTkButton(
            self, text="Back to previous designs",
            command=self.back_to_prev_designs,
            font=("Helvetica", 14),
            fg_color="#3B8ED0",
            hover_color="#1C6EA4"
        )
        self.back_prev_button.grid(row=11, column=0, pady=5)

        self.finish_button = ctk.CTkButton(
            self, text="Finish Designing",
            command=self.finish_designing,
            font=("Helvetica", 16, "bold"),
            fg_color="#111111",
            hover_color="#333333",
            text_color="#FFFFFF",
            corner_radius=8
        )
        self.finish_button.grid(row=12, column=0, columnspan=3, pady=20, sticky="ew")

        self.show_saved_genes()

    def set_mutation_rate(self, value):
        self.mutation_rate = float(value)

    def toggle_gene(self, index, button):
        if index in self.selected_gene_indices:
            self.selected_gene_indices.remove(index)
            button.configure(
                text="Toggle Selection",
                fg_color="gray",
                text_color="black"
            )
        else:
            self.selected_gene_indices.append(index)
            button.configure(
                text="Selected",
                fg_color="lightgreen",
                text_color="black"
            )

    def save_gene(self, index, button):
        gene = self.current_gene_pool[index]
        if gene in self.saved_genes:
            self.saved_genes.remove(gene)
            if gene in self.saved_gene_widgets:
                fig, widget = self.saved_gene_widgets[gene]
                plt.close(fig)
                widget.destroy()
                del self.saved_gene_widgets[gene]
            button.configure(
                text="Save",
                fg_color="gray",
                text_color="black"
            )
        else:
            self.saved_genes.append(gene)
            button.configure(
                text="Saved",
                fg_color="lightgreen",
                text_color="black"
            )
        self.show_saved_genes()

    def submit_selection(self):
        for fig, widget in self.current_gene_pool_figures:
            plt.close(fig)
            widget.destroy()

        # Also close saved design figures
        for fig, widget in self.saved_gene_widgets.values():
            plt.close(fig)
            widget.destroy()
        self.saved_gene_widgets = {}

        if self.selected_gene_indices:
            self.gene_pools_previous.append(self.current_gene_pool)
            selected_genes = [self.current_gene_pool[i] for i in self.selected_gene_indices]
            self.current_gene_pool = breed_new_designs_with_auto_selection(selected_genes, self.mutation_rate)
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
        popup = ctk.CTkToplevel(self)
        popup.title("Design Preview")
        popup.geometry("600x400")
        popup.bind("<FocusOut>", lambda event: popup.destroy())
        content_frame = ctk.CTkFrame(popup, fg_color="#F9FAFB")
        content_frame.pack(expand=True, fill="both", padx=10, pady=10)
        large_fig = self.current_gene_pool[index].render_design(scale=1.75)
        for ax in large_fig.get_axes():
            ax.set_axis_off()
        large_canvas = FigureCanvasTkAgg(large_fig, master=content_frame)
        large_canvas.draw()
        large_canvas.get_tk_widget().pack(expand=True, fill="both", padx=5, pady=5)
        close_button = ctk.CTkButton(content_frame, text="Close", command=popup.destroy)
        close_button.pack(pady=5)
