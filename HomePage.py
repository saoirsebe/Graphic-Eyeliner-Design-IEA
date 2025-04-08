import customtkinter as ctk
import tkinter as tk
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class HomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        # HomePage remains a fixed-size frame
        super().__init__(parent, width=1250, height=650, fg_color="#F9FAFB")
        self.controller = controller
        # List to store tuples (figure, widget) for recent designs
        self.recent_design_figures = []
        # Create an outer container and scrollable frame so that the whole page is scrollable.
        self.outer_scroll = ctk.CTkFrame(self, fg_color="#F9FAFB")
        self.outer_scroll.pack(expand=True, fill="both")
        self.scrollable_main = ctk.CTkScrollableFrame(self.outer_scroll, fg_color="#F9FAFB")
        self.scrollable_main.pack(expand=True, fill="both")
        # Bind mouse scroll events to the scrollable frame.
        self.scrollable_main.bind("<Enter>", self.bind_scroll_events)
        self.scrollable_main.bind("<Leave>", self.unbind_scroll_events)
        self.create_widgets()

    def bind_scroll_events(self, event):
        """Bind mouse scrolling when the cursor enters the scrollable frame."""
        self.scrollable_main.bind_all("<MouseWheel>", self._on_mouse_scroll)  # Windows/macOS
        self.scrollable_main.bind_all("<Button-4>", self._on_mouse_scroll)     # Linux scroll up
        self.scrollable_main.bind_all("<Button-5>", self._on_mouse_scroll)     # Linux scroll down

    def unbind_scroll_events(self, event):
        """Unbind mouse scrolling when the cursor leaves the scrollable frame."""
        self.scrollable_main.unbind_all("<MouseWheel>")
        self.scrollable_main.unbind_all("<Button-4>")
        self.scrollable_main.unbind_all("<Button-5>")

    def _on_mouse_scroll(self, event):
        """Scroll the scrollable frame more on each mouse wheel event."""
        SCROLL_STEP = 20  # Adjust this value for faster or slower scrolling
        if event.num == 4:  # Linux scroll up
            self.scrollable_main._canvas.yview_scroll(-SCROLL_STEP, "units")
        elif event.num == 5:  # Linux scroll down
            self.scrollable_main._canvas.yview_scroll(SCROLL_STEP, "units")
        else:  # Windows/macOS
            self.scrollable_main._canvas.yview_scroll(-SCROLL_STEP if event.delta > 0 else SCROLL_STEP, "units")

    def create_widgets(self):
        # Use the scrollable_main as the parent for all widgets.
        self.scrollable_main.grid_columnconfigure(0, weight=1)
        self.scrollable_main.grid_columnconfigure(1, weight=1)
        self.scrollable_main.grid_columnconfigure(2, weight=1)
        self.scrollable_main.grid_rowconfigure(0, weight=1)
        self.scrollable_main.grid_rowconfigure(1, weight=0)
        self.scrollable_main.grid_rowconfigure(2, weight=0)
        self.scrollable_main.grid_rowconfigure(3, weight=0)
        self.scrollable_main.grid_rowconfigure(4, weight=1)

        # ========== (1) Top Navigation / Logo + Logout ==========
        self.top_nav_frame = ctk.CTkFrame(self.scrollable_main, fg_color="#F9FAFB")
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
            self.scrollable_main,
            text="The Beauty of Eyeliner",
            font=("Helvetica", 40, "bold"),
            text_color="#111111"
        )
        self.main_title.grid(row=1, column=0, columnspan=3, pady=(10, 0))

        self.subtext_label = ctk.CTkLabel(
            self.scrollable_main,
            text="Unleash your creativity with bold, innovative eyeliner designs!",
            font=("Helvetica", 18),
            text_color="#555555"
        )
        self.subtext_label.grid(row=2, column=0, columnspan=3, pady=(0, 20))

        # ========== (3) Central "Start Designing" Button ==========
        self.start_designing_button = ctk.CTkButton(
            self.scrollable_main,
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
        # Left: Recently Saved Designs (smaller height)
        self.recent_outer_frame = ctk.CTkFrame(
            self.scrollable_main,
            fg_color="#C2E7E5",
            corner_radius=20,
            height=40  # Reduced height for the outer box
        )
        self.recent_outer_frame.grid(row=4, column=0, columnspan=1, padx=(50, 10), pady=20, sticky="nsew")
        self.recent_outer_frame.grid_propagate(False)
        self.recent_outer_frame.grid_columnconfigure(0, weight=1)

        self.recent_label = ctk.CTkLabel(
            self.recent_outer_frame,
            text="Your Recently Saved Designs",
            font=("Helvetica", 16, "bold"),
            text_color="#111111"
        )
        self.recent_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.recent_inner_frame = ctk.CTkFrame(
            self.recent_outer_frame,
            fg_color="#FFFFFF",
            corner_radius=10
        )
        self.recent_inner_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.recent_inner_frame.grid_columnconfigure(0, weight=1)

        self.show_recent_designs()

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
        self.example_outer_frame = ctk.CTkFrame(
            self.scrollable_main,
            fg_color="#FBCFD2",
            corner_radius=20,
            width=50
        )
        self.example_outer_frame.grid(row=4, column=2, columnspan=1, padx=(10, 50), pady=20, sticky="nsew")
        self.example_outer_frame.grid_columnconfigure(0, weight=1)

        self.example_label = ctk.CTkLabel(
            self.example_outer_frame,
            text="Example Designs",
            font=("Helvetica", 16, "bold"),
            text_color="#111111"
        )
        self.example_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.example_inner_frame = ctk.CTkFrame(
            self.example_outer_frame,
            fg_color="#FFFFFF",
            corner_radius=10
        )
        self.example_inner_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.example_inner_frame.grid_columnconfigure(0, weight=1)

        self.show_example_designs()

    def show_recent_designs(self):
        """
        Displays a small preview of the user's most recent saved designs inside recent_inner_frame.
        """
        self.cleanup_recent_designs()
        for widget in self.recent_inner_frame.winfo_children():
            widget.destroy()
        all_saved = self.controller.all_saved_designs
        recent_designs = all_saved[-2:]  # up to the last 2 designs
        recent_designs.reverse()         # newest first
        for i, design in enumerate(recent_designs):
            design_frame = ctk.CTkFrame(self.recent_inner_frame, fg_color="#FFFFFF")
            design_frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            fig = design.render_design()
            for ax in fig.get_axes():
                ax.set_axis_off()
            canvas = FigureCanvasTkAgg(fig, master=design_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(expand=True, fill="both")
            plt.close(fig)
            self.recent_design_figures.append(canvas_widget)
            canvas_widget.bind("<Button-1>", lambda event, d=design: self.show_design_popup(d))

    def cleanup_recent_designs(self):
        """Closes all matplotlib figures and destroys their widgets for recent designs."""
        for widget in self.recent_design_figures:
            widget.destroy()
        self.recent_design_figures = []

    def show_example_designs(self):
        """
        Displays example design images ("Design1.png" and "Design2.png") in example_inner_frame.
        """
        from PIL import Image
        for widget in self.example_inner_frame.winfo_children():
            widget.destroy()
        design_images = ["Design1.png", "Design2.png"]
        self.example_design_images = []
        for i, img_filename in enumerate(design_images):
            try:
                img = Image.open(img_filename)
            except Exception as e:
                print(f"Error loading image '{img_filename}': {e}")
                continue
            ctk_img = ctk.CTkImage(light_image=img, size=(200, 200))
            self.example_design_images.append(ctk_img)
            image_label = ctk.CTkLabel(self.example_inner_frame, image=ctk_img, text="")
            image_label.grid(row=i, column=0, padx=10, pady=5, sticky="nsew")

    def show_design_popup(self, design):
        """
        Opens a popup window to display the selected design at a larger scale.
        """
        popup = ctk.CTkToplevel(self)
        popup.title("Design Preview")
        popup.geometry("500x400")
        content_frame = ctk.CTkFrame(popup, fg_color="#F9FAFB")
        content_frame.pack(expand=True, fill="both", padx=5, pady=5)
        large_fig = design.render_design(scale=1.75)
        for ax in large_fig.get_axes():
            ax.set_axis_off()
        large_canvas = FigureCanvasTkAgg(large_fig, master=content_frame)
        large_canvas.draw()
        large_canvas.get_tk_widget().pack(expand=True, fill="both", padx=5, pady=5)
        close_button = ctk.CTkButton(content_frame, text="Close", command=popup.destroy)
        close_button.pack(pady=5)

    def logout(self):
        # Cleanup recent design figures and inner frame before leaving HomePage.
        self.cleanup_recent_designs()
        for widget in self.recent_inner_frame.winfo_children():
            widget.destroy()
        self.controller.show_page("LoginPage")
