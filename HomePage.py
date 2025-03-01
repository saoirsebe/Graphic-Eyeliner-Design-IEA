import customtkinter as ctk

class HomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, width=1200, height=600)
        self.grid_propagate(False)
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.label_title = ctk.CTkLabel(self, text="Welcome to the Eyeliner Design App", font=("Arial", 24))
        self.label_title.grid(row=0, column=0, columnspan=2, pady=30)

        self.greeting_label = ctk.CTkLabel(self, text="Hello, User!", font=("Arial", 18))
        self.greeting_label.grid(row=1, column=0, columnspan=2, pady=10)

        self.start_button = ctk.CTkButton(self, text="Go to Design Page", command=lambda: self.controller.show_page("DesignPage"), width=200, height=50)
        self.start_button.grid(row=2, column=0, columnspan=2, pady=20)

        self.logout_button = ctk.CTkButton(self, text="Logout", command=self.logout, width=200, height=50, fg_color="red", hover_color="#8B0000")
        self.logout_button.grid(row=3, column=0, columnspan=2, pady=20)

    def logout(self):
        self.controller.show_page("LoginPage")
