import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class HomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, width=1250, height=650, fg_color="#F9FAFB")
        self.grid_propagate(False)
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        # Grid configuration for the main layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)
        self.grid_rowconfigure(4, weight=1)

        # ========== (1) Top Navigation / Logo + Logout ==========
        self.top_nav_frame = ctk.CTkFrame(self, fg_color="#F9FAFB")
        self.top_nav_frame.grid(row=0, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        self.top_nav_frame.grid_columnconfigure(0, weight=1)
        self.top_nav_frame.grid_columnconfigure(1, weight=1)
        self.top_nav_frame.grid_columnconfigure(2, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.top_nav_frame,
            text="EyelinerApp",
            font=("Helvetica", 18, "bold"),
            text_color="#111111"
        )
        self.logo_label.grid(row=0, column=0, sticky="w", padx=(30, 0))

        self.logout_button = ctk.CTkButton(
            self.top_nav_frame,
            text="Logout",
            command=self.logout,
            fg_color="#D9534F",
            hover_color="#C9302C",
            font=("Helvetica", 14)
        )
        self.logout_button.grid(row=0, column=2, sticky="e", padx=(0, 30))

        # ========== (2) Big Title & Subheading ==========
        self.main_title = ctk.CTkLabel(
            self,
            text="The Beauty of Eyeliner",
            font=("Helvetica", 40, "bold"),
            text_color="#111111"
        )
        self.main_title.grid(row=1, column=0, columnspan=3, pady=(10, 0))

        self.subtext_label = ctk.CTkLabel(
            self,
            text="Unleash your creativity with bold, innovative eyeliner designs!",
            font=("Helvetica", 18),
            text_color="#555555"
        )
        self.subtext_label.grid(row=2, column=0, columnspan=3, pady=(0, 20))

        # ========== (3) Central "Start Designing" Button ==========
        self.start_designing_button = ctk.CTkButton(
            self,
            text="Start Designing",
            command=lambda: self.controller.show_page("DesignPage"),
            width=220,
            height=60,
            fg_color="#111111",
            hover_color="#333333",
            text_color="#FFFFFF",
            font=("Helvetica", 16, "bold"),
            corner_radius=8
        )
        self.start_designing_button.grid(row=3, column=0, columnspan=3, pady=10)

        # ========== (4) Two Colored Boxes for Saved/Example Designs ==========
        # Left: Recently Saved Designs
        self.recent_outer_frame = ctk.CTkFrame(self, fg_color="#C2E7E5", corner_radius=20)  # Teal-like color
        self.recent_outer_frame.grid(row=4, column=0, columnspan=1, padx=(50, 10), pady=20, sticky="nsew")
        self.recent_outer_frame.grid_columnconfigure(0, weight=1)

        # Title text on the colored area
        self.recent_label = ctk.CTkLabel(
            self.recent_outer_frame,
            text="Your Recently Saved Designs",
            font=("Helvetica", 16, "bold"),
            text_color="#111111"
        )
        self.recent_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        # White inner frame for the actual design previews
        self.recent_inner_frame = ctk.CTkFrame(self.recent_outer_frame, fg_color="#FFFFFF", corner_radius=10)
        self.recent_inner_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.recent_inner_frame.grid_columnconfigure(0, weight=1)

        # Show the last few saved designs
        self.show_recent_designs()

        # Button to go to saved designs page
        self.view_all_button = ctk.CTkButton(
            self.recent_outer_frame,
            text="View All Saved Designs",
            command=lambda: self.controller.show_page("SaveDesignPage"),
            fg_color="#3B8ED0",
            hover_color="#1C6EA4",
            font=("Helvetica", 14)
        )
        self.view_all_button.grid(row=2, column=0, pady=(0, 10))

        # Right: Example Designs
        self.example_outer_frame = ctk.CTkFrame(self, fg_color="#FBCFD2", corner_radius=20)  # Light pink color
        self.example_outer_frame.grid(row=4, column=2, columnspan=1, padx=(10, 50), pady=20, sticky="nsew")
        self.example_outer_frame.grid_columnconfigure(0, weight=1)

        self.example_label = ctk.CTkLabel(
            self.example_outer_frame,
            text="Example Designs",
            font=("Helvetica", 16, "bold"),
            text_color="#111111"
        )
        self.example_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        # White inner frame for the example designs
        self.example_inner_frame = ctk.CTkFrame(self.example_outer_frame, fg_color="#FFFFFF", corner_radius=10)
        self.example_inner_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.example_inner_frame.grid_columnconfigure(0, weight=1)

        # Show example designs (replace placeholders with real data if you like)
        self.show_example_designs()

    def show_recent_designs(self):
        """
        Displays a small preview of the user's most recent saved designs
        inside the recent_inner_frame.
        """
        # Clear anything old
        for widget in self.recent_inner_frame.winfo_children():
            widget.destroy()

        # Get the last 3 designs from the global saved designs (if fewer than 3 exist, show all).
        all_saved = self.controller.all_saved_designs
        recent_designs = all_saved[-2:]  # up to the last 3
        recent_designs.reverse()         # so the newest is at the top

        # Display each design in a small frame
        for i, design in enumerate(recent_designs):
            design_frame = ctk.CTkFrame(self.recent_inner_frame, fg_color="#FFFFFF")
            design_frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")

            # Render the design at a smaller scale
            fig = design.render_design()#scale=0.5)
            for ax in fig.get_axes():
                ax.set_axis_off()
            canvas = FigureCanvasTkAgg(fig, master=design_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(expand=True, fill="both")

            # Optional: on click, show a popup with a larger preview
            canvas_widget.bind("<Button-1>", lambda event, d=design: self.show_design_popup(d))


    def show_example_designs(self):
        """
        Displays some placeholder example designs in the example_inner_frame.
        Replace with real designs or images as desired.
        """
        # Clear anything old
        for widget in self.example_inner_frame.winfo_children():
            widget.destroy()

        example_names = ["Winged Classic", "Bold Cat Eye", "Graphic Liner"]
        for i, name in enumerate(example_names):
            label = ctk.CTkLabel(
                self.example_inner_frame,
                text=name,
                font=("Helvetica", 14),
                text_color="#333333"
            )
            label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

    def show_design_popup(self, design):
        """
        Opens a popup window to display the selected design at a larger scale.
        """
        popup = ctk.CTkToplevel(self)
        popup.title("Design Preview")
        popup.geometry("500x400")

        content_frame = ctk.CTkFrame(popup, fg_color="#F9FAFB")
        content_frame.pack(expand=True, fill="both", padx=5, pady=5)

        large_fig = design.render_design(scale=2)
        for ax in large_fig.get_axes():
            ax.set_axis_off()
        large_canvas = FigureCanvasTkAgg(large_fig, master=content_frame)
        large_canvas.draw()
        large_canvas.get_tk_widget().pack(expand=True, fill="both", padx=5, pady=5)

        close_button = ctk.CTkButton(content_frame, text="Close", command=popup.destroy)
        close_button.pack(pady=5)

    def logout(self):
        self.controller.show_page("LoginPage")
