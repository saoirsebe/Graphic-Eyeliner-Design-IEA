# SaveDesignPage.py
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from AnalyseDesign import analyse_negative


class SaveDesignPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, width=1200, height=600)
        self.grid_propagate(False)
        for i in range(6):
            self.grid_columnconfigure(i, weight=1)

        self.controller = controller

        # Title label
        self.label_title = ctk.CTkLabel(self, text="Saved Designs", font=("Arial", 24))
        self.label_title.pack(pady=10)

        # Create a scrollable frame to hold the saved designs
        self.scrollable_frame = ctk.CTkScrollableFrame(self, fg_color="#2A2A2A")
        self.scrollable_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Back button to return to the design page
        self.back_button = ctk.CTkButton(self, text="Back to Homepage",width=200, height=50,
                                         command=lambda: controller.show_page("HomePage"))
        self.back_button.pack(pady=10)

        self.update_designs()

    def update_designs(self):
        # Clear any existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        saved_designs = self.controller.all_saved_designs  # Global list maintained in App
        num_columns = 3

        # Layout each saved design in a grid
        for i, design in enumerate(saved_designs):
            segments = design.get_all_nodes()
            for segment in segments:
                print(f"segment colour {segment.colour} points_ array:", segment.points_array)
            neg = analyse_negative(design)
            print("Negative score:",neg)

            frame = ctk.CTkFrame(self.scrollable_frame)
            frame.grid(row=i // num_columns, column=i % num_columns, padx=10, pady=10, sticky="nsew")

            # Render the design (assuming design.render_design() returns a matplotlib Figure)
            fig = design.render_design()
            for ax in fig.get_axes():
                ax.set_axis_off()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(expand=True, fill="both")

            # Bind click event to show a larger popup version
            canvas_widget.bind("<Button-1>", lambda event, d=design: self.show_design_popup(d))

    def show_design_popup(self, design):
        popup = ctk.CTkToplevel(self)
        popup.title("Design Preview")
        popup.geometry("800x600")

        content_frame = ctk.CTkFrame(popup)
        content_frame.pack(expand=True, fill="both", padx=10, pady=10)

        large_fig = design.render_design(scale=2)  # Render with a larger scale
        large_canvas = FigureCanvasTkAgg(large_fig, master=content_frame)
        large_canvas.draw()
        large_canvas.get_tk_widget().pack(expand=True, fill="both", padx=5, pady=5)

        close_button = ctk.CTkButton(content_frame, text="Close", command=popup.destroy)
        close_button.pack(pady=5)

