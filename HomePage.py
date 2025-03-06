import customtkinter as ctk

class HomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        # Set a custom background color for a modern look
        super().__init__(parent, width=1250, height=650, fg_color="#1E1E2E")
        self.grid_propagate(False)
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        # Configure the grid so the central column expands for balanced content.
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_columnconfigure(2, weight=1)

        # Title label with a bold, fun font and a bright text color
        self.label_title = ctk.CTkLabel(
            self,
            text="Welcome to the Eyeliner Design App",
            font=("Helvetica", 32, "bold"),
            text_color="#FFFFFF"
        )
        self.label_title.grid(row=0, column=0, columnspan=3, pady=(30, 10))

        # Tagline to add personality
        self.label_tagline = ctk.CTkLabel(
            self,
            text="Unleash your creativity with fun, innovative eyeliner designs!",
            font=("Helvetica", 20),
            text_color="#AAAAAA"
        )
        self.label_tagline.grid(row=1, column=0, columnspan=3, pady=(0, 20))

        # Greeting label for a personal touch
        self.greeting_label = ctk.CTkLabel(
            self,
            text="Hello, User!",
            font=("Helvetica", 24),
            text_color="#FFFFFF"
        )
        self.greeting_label.grid(row=2, column=0, columnspan=3, pady=10)

        # A frame to contain the buttons, with a subtle contrasting background and rounded corners
        self.buttons_frame = ctk.CTkFrame(self, fg_color="#2E2E3E", corner_radius=15)
        self.buttons_frame.grid(row=3, column=0, columnspan=3, pady=20, padx=20, sticky="nsew")
        self.buttons_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="group1")

        # Button to navigate to the Design Page
        self.design_page_button = ctk.CTkButton(
            self.buttons_frame,
            text="Go to Design Page",
            command=lambda: self.controller.show_page("DesignPage"),
            width=200,
            height=50,
            fg_color="#3B8ED0",
            hover_color="#1C6EA4",
            font=("Helvetica", 16)
        )
        self.design_page_button.grid(row=0, column=0, padx=10, pady=20)

        # Button to view saved designs
        self.saved_page_button = ctk.CTkButton(
            self.buttons_frame,
            text="Saved Designs",
            command=lambda: self.controller.show_page("SaveDesignPage"),
            width=200,
            height=50,
            fg_color="#3B8ED0",
            hover_color="#1C6EA4",
            font=("Helvetica", 16)
        )
        self.saved_page_button.grid(row=0, column=1, padx=10, pady=20)

        # Logout button, styled in a warm red to indicate its importance
        self.logout_button = ctk.CTkButton(
            self.buttons_frame,
            text="Logout",
            command=self.logout,
            width=200,
            height=50,
            fg_color="#D9534F",
            hover_color="#C9302C",
            font=("Helvetica", 16)
        )
        self.logout_button.grid(row=0, column=2, padx=10, pady=20)

    def logout(self):
        self.controller.show_page("LoginPage")
