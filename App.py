import customtkinter as ctk

from DesignPage import DesignPage
from DualScrollableFrame import DualScrollableFrame
from HomePage import HomePage
from LoginPage import LoginPage, SignUpPage


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")  # Options: "System", "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        self.title("Eyeliner Design App")
        self.geometry("1200x600")
        self.configure(bg="#1E1E1E")

        # Configure the main grid so the scrollable frame fills the entire window
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create a scrollable frame using grid
        self.scrollable_frame = ctk.CTkScrollableFrame(self, fg_color="#2A2A2A", corner_radius=10)
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        # Ensure the scrollable frame's internal grid expands as well
        self.scrollable_frame.grid_rowconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        self.page_classes = {
            "HomePage": HomePage,
            "DesignPage": DesignPage,
            "LoginPage": LoginPage,
            "SignUpPage": SignUpPage,
        }

        self.pages = {}  # Store pages dynamically
        for PageClass in (HomePage, DesignPage, SignUpPage, LoginPage):
            page_name = PageClass.__name__
            page = PageClass(parent=self.scrollable_frame, controller=self)
            self.pages[page_name] = page
            # Use grid to fill the available space
            page.grid(row=0, column=0, sticky="nsew")
            # Configure the scrollable frame to let the page expand
            self.scrollable_frame.grid_rowconfigure(0, weight=1)
            self.scrollable_frame.grid_columnconfigure(0, weight=1)

        self.show_page("DesignPage")

    def show_page(self, page_name):
        """Show the selected page."""
        for widget in self.scrollable_frame.winfo_children():
            widget.grid_forget()
        self.pages[page_name].grid(row=0, column=0, sticky="nsew")
        # Re-configure grid weights to ensure expansion
        self.scrollable_frame.grid_rowconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)


if __name__ == "__main__":
    app = App()
    app.mainloop()
