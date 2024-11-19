import tkinter as tk
from tkinter import messagebox
from Page import Page
class HomePage(Page):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_widgets()

    def create_widgets(self):
        # Grid configuration for HomePage layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)

        # Title label
        tk.Label(self, text="Welcome to the Eyeliner Design App", font=("Arial", 18)).grid(row=0, column=0, columnspan=2, pady=30)

        # Greeting message (optional: add username if you want a personalized greeting)
        # You can use a global variable or passed user information here for personalization
        greeting_label = tk.Label(self, text="Hello, User!", font=("Arial", 14))
        greeting_label.grid(row=1, column=0, columnspan=2, pady=10)

        # Start Designing Button
        tk.Button(self, text="Go to design page", command=lambda: self.controller.show_page("DesignPage"), width=20, height=2).grid(row=2, column=0, columnspan=2, pady=20)

        # Logout Button
        tk.Button(self, text="Logout", command=self.logout, width=20, height=2, bg="red", fg="white").grid(row=3, column=0, columnspan=2, pady=10)

    def logout(self):
        # You can add additional logic here for logging out if needed
        self.controller.show_page("LoginPage")