import tkinter as tk
from tkinter import messagebox
from Page import Page
USERS_FILE = "users.json"
import json
import os
import bcrypt

def load_users():
    """Load users from the JSON file."""
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    """Save users to the JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    """Hash the password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(stored_hash, password):
    """Check if the entered password matches the stored hash."""
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))

class LoginPage(Page):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_widgets()
        self.users = load_users()

    def create_widgets(self):
        # Grid configuration for LoginPage layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)

        # Title label
        tk.Label(self, text="Login", font=("Arial", 24)).grid(row=0, column=0, columnspan=2, pady=20)

        # Username Label and Entry
        tk.Label(self, text="Username").grid(row=1, column=0, sticky="e", padx=10, pady=5)
        username_entry = tk.Entry(self)
        username_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Password Label and Entry
        tk.Label(self, text="Password").grid(row=2, column=0, sticky="e", padx=10, pady=5)
        password_entry = tk.Entry(self, show="*")
        password_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Login function
        def login():
            username = username_entry.get()
            password = password_entry.get()
            if username in self.users and check_password(self.users[username], password):
                self.controller.show_page("HomePage")
            else:
                messagebox.showerror("Login Failed", "Invalid username or password")

        # Go to SignUp page function
        def go_to_signup():
            self.controller.show_page("SignUpPage")

        # Login and Signup buttons
        tk.Button(self, text="Login", command=login).grid(row=3, column=0, columnspan=2, pady=10)
        tk.Button(self, text="Sign Up", command=go_to_signup).grid(row=4, column=0, columnspan=2, pady=5)


# SignUpPage
class SignUpPage(Page):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.users = load_users()
        self.create_widgets()

    def create_widgets(self):
        # Grid configuration for SignUpPage layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_rowconfigure(5, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)

        # Title label
        tk.Label(self, text="Sign Up", font=("Arial", 24)).grid(row=0, column=0, columnspan=2, pady=20)

        # Username Label and Entry
        tk.Label(self, text="Username").grid(row=1, column=0, sticky="e", padx=10, pady=5)
        username_entry = tk.Entry(self)
        username_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Password Label and Entry
        tk.Label(self, text="Password").grid(row=2, column=0, sticky="e", padx=10, pady=5)
        password_entry = tk.Entry(self, show="*")
        password_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Confirm Password Label and Entry
        tk.Label(self, text="Confirm Password").grid(row=3, column=0, sticky="e", padx=10, pady=5)
        confirm_password_entry = tk.Entry(self, show="*")
        confirm_password_entry.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        # SignUp function
        def signup():
            username = username_entry.get()
            password = password_entry.get()
            confirm_password = confirm_password_entry.get()

            if username in self.users:
                messagebox.showerror("Error", "Username already exists")
            elif password != confirm_password:
                messagebox.showerror("Error", "Passwords do not match")
            elif not username or not password:
                messagebox.showerror("Error", "Fields cannot be empty")
            else:
                self.users[username] = hash_password(password)
                save_users(self.users)
                messagebox.showinfo("Success", "Account created successfully")
                self.controller.show_page("LoginPage")

        # SignUp and Back to Login buttons
        tk.Button(self, text="Sign Up", command=signup).grid(row=4, column=0, columnspan=2, pady=10)
        tk.Button(self, text="Back to Login", command=lambda: self.controller.show_page("LoginPage")).grid(row=5, column=0, columnspan=2, pady=5)




