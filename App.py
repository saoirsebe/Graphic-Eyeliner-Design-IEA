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
        ctk.set_appearance_mode("dark")  # Options: "System", "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        self.title("Eyeliner Design App")
        self.geometry("1200x600")
        self.configure(bg="#1E1E1E")

        #Global list to store all saved designs from all generations
        self.all_saved_designs = []

        # Create a container frame that fills the entire window
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True, padx=10, pady=10)

        # Use grid inside container so pages can expand dynamically.
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.page_classes = {
            "HomePage": HomePage,
            "DesignPage": DesignPage,
            "LoginPage": LoginPage,
            "SignUpPage": SignUpPage,
            "SaveDesignPage": SaveDesignPage,
        }

        self.pages = {}  # Dictionary to store each page's dual scrollable frame

        for PageClass in (DesignPage, SignUpPage, LoginPage):
            page_name = PageClass.__name__
            dual_frame = DualScrollableFrame(self.container, fg_color="#2A2A2A")
            dual_frame.grid(row=0, column=0, sticky="nsew")
            dual_frame.grid_rowconfigure(0, weight=1)
            dual_frame.grid_columnconfigure(0, weight=1)

            page = PageClass(parent=dual_frame.inner_frame, controller=self)
            page.grid(row=0, column=0, sticky="nsew")
            page.grid_rowconfigure(0, weight=1)
            page.grid_columnconfigure(0, weight=1)

            self.pages[page_name] = dual_frame

        # For SaveDesignPage and HomePage, instantiate it directly:
        save_design_page = SaveDesignPage(parent=self.container, controller=self)
        save_design_page.grid(row=0, column=0, sticky="nsew")
        self.pages["SaveDesignPage"] = save_design_page

        home_page = HomePage(parent=self.container, controller=self)
        home_page.grid(row=0, column=0, sticky="nsew")
        self.pages["HomePage"] = home_page

        self.show_page("DesignPage")
        # Optionally bind container resize event to update page size if needed.
        self.container.bind("<Configure>", self._on_container_configure)

    def _on_container_configure(self, event):
        # When container resizes, force each page's frame to use the new size.
        for frame in self.pages.values():
            frame.configure(width=event.width, height=event.height)

    def show_page(self, page_name):
        """Show the selected page by hiding all others."""
        for widget in self.container.winfo_children():
            widget.grid_forget()
        for page in self.pages.values():
            page.grid_remove()  # or pack_forget() depending on your layout
        self.pages[page_name].grid(row=0, column=0, sticky="nsew")
        if page_name == "SaveDesignPage" and hasattr(self.pages[page_name], "update_designs"):
            self.pages[page_name].update_designs()

    def add_saved_designs(self, designs):
        # Add new saved designs to the global list (avoid duplicates)
        for design in designs:
            if design not in self.all_saved_designs:
                self.all_saved_designs.append(design)
                #print("saved designs:",self.all_saved_designs)

    def save_user_designs(self,username, designs):
        """Save the user's designs to a file using pickle."""
        filename = f"{username}_designs.pkl"
        with open(filename, "wb") as f:
            pickle.dump(designs, f)

    def load_user_designs(self,username):
        """Load the user's designs from a pickle file."""
        filename = f"{username}_designs.pkl"
        try:
            with open(filename, "rb") as f:
                designs = pickle.load(f)
            return designs
        except FileNotFoundError:
            return []


if __name__ == "__main__":
    app = App()
    app.mainloop()
