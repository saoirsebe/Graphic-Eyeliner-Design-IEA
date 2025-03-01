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
        }

        self.pages = {}  # Dictionary to store each page's dual scrollable frame

        for PageClass in (HomePage, DesignPage, SignUpPage, LoginPage):
            page_name = PageClass.__name__
            # Create a new dual scrollable frame for this page
            dual_frame = DualScrollableFrame(self.container, fg_color="#2A2A2A")
            dual_frame.grid(row=0, column=0, sticky="nsew")
            dual_frame.grid_rowconfigure(0, weight=1)
            dual_frame.grid_columnconfigure(0, weight=1)
            # Do not set fixed size; let it grow with the container.
            # dual_frame.configure(width=1200, height=600)
            # dual_frame.grid_propagate(False)

            # Initialize the page inside the dual scrollable frame's inner_frame.
            page = PageClass(parent=dual_frame.inner_frame, controller=self)
            page.grid(row=0, column=0, sticky="nsew")
            page.grid_rowconfigure(0, weight=1)
            page.grid_columnconfigure(0, weight=1)

            self.pages[page_name] = dual_frame

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
        self.pages[page_name].grid(row=0, column=0, sticky="nsew")


if __name__ == "__main__":
    app = App()
    app.mainloop()
