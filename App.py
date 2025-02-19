import customtkinter as ctk

from DesignPage import DesignPage
from HomePage import HomePage
from LoginPage import LoginPage, SignUpPage


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")  # Options: "System", "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        self.title("Eyeliner Design App")
        self.geometry("1000x600")
        self.configure(bg="#1E1E1E")

        # Create a scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self, fg_color="#2A2A2A", corner_radius=10)
        self.scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)

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
            page.pack(fill="both", expand=True)

        self.show_page("DesignPage")

    def show_page(self, page_name):
        """Show the selected page."""
        for widget in self.scrollable_frame.winfo_children():
            widget.pack_forget()
        self.pages[page_name].pack(fill="both", expand=True)


if __name__ == "__main__":
    app = App()
    app.mainloop()
