import customtkinter as ctk
from DesignPage import DesignPage
from DualScrollableFrame import DualScrollableFrame
from HomePage import HomePage
from LoginPage import LoginPage, SignUpPage
from SavedDesigns import SaveDesignPage
import pickle


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # Use a light appearance mode and a built-in color theme.
        ctk.set_appearance_mode("light")   # Options: "system", "light", "dark"
        ctk.set_default_color_theme("blue")  # or "green", "dark-blue", etc.

        self.title("Eyeliner Design App")
        self.geometry("1200x600")

        # Give the main window a pastel background (or white if you prefer).
        self.configure(bg="#F9FAFB")  # Light pastel grayish background

        # Global list to store all saved designs from all generations
        self.all_saved_designs = []

        # Create a container frame that fills the entire window
        # Use a light pastel background and rounded corners for a modern aesthetic.
        self.container = ctk.CTkFrame(self, fg_color="#F9FAFB", corner_radius=0)
        self.container.pack(fill="both", expand=True)

        # Use grid inside the container so pages can expand dynamically
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.page_classes = {
            "HomePage": HomePage,
            "DesignPage": DesignPage,
            "LoginPage": LoginPage,
            "SignUpPage": SignUpPage,
            "SaveDesignPage": SaveDesignPage,
        }

        self.pages = {}  # Dictionary to store each page widget (or DualScrollableFrame)

        # Create pages that use DualScrollableFrame
        for PageClass in (DesignPage, SignUpPage, LoginPage):
            page_name = PageClass.__name__
            # Create a pastel, rounded DualScrollableFrame
            dual_frame = DualScrollableFrame(
                self.container,
                fg_color="#FFFFFF",      # White background for the outer frame
                corner_radius=15
            )
            dual_frame.grid(row=0, column=0, sticky="nsew")
            dual_frame.grid_rowconfigure(0, weight=1)
            dual_frame.grid_columnconfigure(0, weight=1)

            # Create the actual page within the dual_frame's inner_frame
            page = PageClass(parent=dual_frame.inner_frame, controller=self)
            page.grid(row=0, column=0, sticky="nsew")
            page.grid_rowconfigure(0, weight=1)
            page.grid_columnconfigure(0, weight=1)

            self.pages[page_name] = dual_frame

        # Create pages that do NOT use DualScrollableFrame
        save_design_page = SaveDesignPage(parent=self.container, controller=self)
        save_design_page.grid(row=0, column=0, sticky="nsew")
        self.pages["SaveDesignPage"] = save_design_page

        home_page = HomePage(parent=self.container, controller=self)
        home_page.grid(row=0, column=0, sticky="nsew")
        self.pages["HomePage"] = home_page

        # Initially show the LoginPage
        self.show_page("LoginPage")

        # Optionally bind container resize event to update page size if needed
        self.container.bind("<Configure>", self._on_container_configure)
        self.all_saved_simulation_runs = []

    def _on_container_configure(self, event):
        # When the container resizes, force each page's frame to use the new size
        for frame in self.pages.values():
            frame.configure(width=event.width, height=event.height)

    def show_page(self, page_name):
        """Show the selected page by hiding all others."""
        # Hide all pages
        for widget in self.container.winfo_children():
            widget.grid_forget()

        # Then grid only the requested page
        self.pages[page_name].grid(row=0, column=0, sticky="nsew")

        # If SaveDesignPage is shown, update designs
        if page_name == "SaveDesignPage" and hasattr(self.pages[page_name], "update_designs"):
            self.pages[page_name].update_designs()

    def add_saved_designs(self, designs):
        # Add new saved designs to the global list (avoid duplicates)
        for design in designs:
            if design not in self.all_saved_designs:
                self.all_saved_designs.append(design)

    def save_user_designs(self, username, designs):
        """Save the user's designs to a file using pickle."""
        filename = f"{username}_designs.pkl"
        with open(filename, "wb") as f:
            pickle.dump(designs, f)

    def load_user_designs(self, username):
        """Load the user's designs from a pickle file."""
        filename = f"{username}_designs.pkl"
        try:
            with open(filename, "rb") as f:
                designs = pickle.load(f)
            return designs
        except FileNotFoundError:
            return []

    def add_saved_simulation_runs(self, runs):
        """
        Add new simulation runs to the global list (avoid duplicates).
        Each run is a tuple: (generations, average_scores).
        """
        for run in runs:
            if run not in self.all_saved_simulation_runs:
                self.all_saved_simulation_runs.append(run)


    def save_user_simulation_runs(self, username, simulation_runs):
        """Save the user's simulation runs to a file using pickle."""
        filename = f"{username}_simulation_runs.pkl"
        with open(filename, "wb") as f:
            pickle.dump(simulation_runs, f)


if __name__ == "__main__":
    app = App()
    app.mainloop()
