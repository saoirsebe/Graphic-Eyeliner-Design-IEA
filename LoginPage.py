import customtkinter as ctk
from tkinter import messagebox
import json
import os
import bcrypt
from Page import Page  # Assumed base page class

USERS_FILE = "users.json"

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

class LoginPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        # Light background for a modern, minimal aesthetic
        super().__init__(parent, width=1200, height=600, fg_color="#F9FAFB")
        self.grid_propagate(False)
        self.controller = controller
        self.users = load_users()
        self.create_widgets()

    def create_widgets(self):
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ---------- Top Bar / Title ----------
        # A top frame to hold a big heading or "logo"
        self.top_frame = ctk.CTkFrame(self, fg_color="#F9FAFB")
        self.top_frame.grid(row=0, column=0, sticky="nsew")
        self.top_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.top_frame,
            text="Welcome Back",
            font=("Helvetica", 32, "bold"),
            text_color="#111111"
        )
        self.title_label.grid(row=0, column=0, pady=(30, 10))

        self.subtitle_label = ctk.CTkLabel(
            self.top_frame,
            text="Log in to continue",
            font=("Helvetica", 16),
            text_color="#555555"
        )
        self.subtitle_label.grid(row=1, column=0, pady=(0, 20))

        # ---------- Main Login Frame ----------
        # A pastel-colored outer frame with rounded corners
        self.login_frame = ctk.CTkFrame(self, fg_color="#C2E7E5", corner_radius=20)
        self.login_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # A white inner frame to hold the actual login widgets
        self.login_inner_frame = ctk.CTkFrame(self.login_frame, fg_color="#FFFFFF", corner_radius=10)
        self.login_inner_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Username label and entry
        self.username_label = ctk.CTkLabel(
            self.login_inner_frame,
            text="Username",
            font=("Helvetica", 14),
            text_color="#111111"
        )
        self.username_label.grid(row=0, column=0, sticky="e", padx=(0,10), pady=5)
        self.username_entry = ctk.CTkEntry(self.login_inner_frame)
        self.username_entry.grid(row=0, column=1, padx=(0,10), pady=5, sticky="ew")

        # Password label and entry
        self.password_label = ctk.CTkLabel(
            self.login_inner_frame,
            text="Password",
            font=("Helvetica", 14),
            text_color="#111111"
        )
        self.password_label.grid(row=1, column=0, sticky="e", padx=(0,10), pady=5)
        self.password_entry = ctk.CTkEntry(self.login_inner_frame, show="*")
        self.password_entry.grid(row=1, column=1, padx=(0,10), pady=5, sticky="ew")

        # Bind Enter key to login
        self.username_entry.bind("<Return>", lambda event: login())
        self.password_entry.bind("<Return>", lambda event: login())

        # Define the login function
        def login():
            username = self.username_entry.get()
            password = self.password_entry.get()
            # Reload users in case new account was added
            self.users = load_users()
            if username in self.users and check_password(self.users[username], password):
                self.controller.current_user = username
                # Load the user's saved designs from file
                self.controller.all_saved_designs = self.controller.load_user_designs(username)
                self.username_entry.delete(0, "end")
                self.password_entry.delete(0, "end")
                self.controller.pages["HomePage"].show_recent_designs()
                self.controller.show_page("HomePage")
            else:
                messagebox.showerror("Login Failed", "Invalid username or password")
                self.password_entry.delete(0, "end")

        # Go to SignUp page function
        def go_to_signup():
            self.controller.show_page("SignUpPage")

        # ---------- Buttons ----------
        self.login_button = ctk.CTkButton(
            self.login_inner_frame,
            text="Login",
            command=login,
            width=200,
            height=40,
            fg_color="#111111",
            hover_color="#333333",
            text_color="#FFFFFF",
            corner_radius=8
        )
        self.login_button.grid(row=2, column=0, columnspan=2, pady=(20,10))

        self.signup_button = ctk.CTkButton(
            self.login_inner_frame,
            text="Sign Up",
            command=go_to_signup,
            width=200,
            height=40,
            fg_color="#3B8ED0",
            hover_color="#1C6EA4",
            text_color="#FFFFFF",
            corner_radius=8
        )
        self.signup_button.grid(row=3, column=0, columnspan=2, pady=5)

class SignUpPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        # Light background for consistency
        super().__init__(parent, width=1200, height=600, fg_color="#F9FAFB")
        self.grid_propagate(False)
        self.controller = controller
        self.users = load_users()
        self.create_widgets()

    def create_widgets(self):
        # Grid configuration
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ---------- Top Bar / Title ----------
        self.top_frame = ctk.CTkFrame(self, fg_color="#F9FAFB")
        self.top_frame.grid(row=0, column=0, sticky="nsew")
        self.top_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.top_frame,
            text="Create Account",
            font=("Helvetica", 32, "bold"),
            text_color="#111111"
        )
        self.title_label.grid(row=0, column=0, pady=(30, 10))

        self.subtitle_label = ctk.CTkLabel(
            self.top_frame,
            text="Join us and start designing!",
            font=("Helvetica", 16),
            text_color="#555555"
        )
        self.subtitle_label.grid(row=1, column=0, pady=(0, 20))

        # ---------- Main Sign-Up Frame ----------
        self.signup_frame = ctk.CTkFrame(self, fg_color="#C2E7E5", corner_radius=20)
        self.signup_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        self.signup_inner_frame = ctk.CTkFrame(self.signup_frame, fg_color="#FFFFFF", corner_radius=10)
        self.signup_inner_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Username
        self.username_label = ctk.CTkLabel(
            self.signup_inner_frame,
            text="Username",
            font=("Helvetica", 14),
            text_color="#111111"
        )
        self.username_label.grid(row=0, column=0, sticky="e", padx=(0,10), pady=5)
        self.username_entry = ctk.CTkEntry(self.signup_inner_frame)
        self.username_entry.grid(row=0, column=1, padx=(0,10), pady=5, sticky="ew")

        # Password
        self.password_label = ctk.CTkLabel(
            self.signup_inner_frame,
            text="Password",
            font=("Helvetica", 14),
            text_color="#111111"
        )
        self.password_label.grid(row=1, column=0, sticky="e", padx=(0,10), pady=5)
        self.password_entry = ctk.CTkEntry(self.signup_inner_frame, show="*")
        self.password_entry.grid(row=1, column=1, padx=(0,10), pady=5, sticky="ew")

        # Confirm Password
        self.confirm_password_label = ctk.CTkLabel(
            self.signup_inner_frame,
            text="Confirm Password",
            font=("Helvetica", 14),
            text_color="#111111"
        )
        self.confirm_password_label.grid(row=2, column=0, sticky="e", padx=(0,10), pady=5)
        self.confirm_password_entry = ctk.CTkEntry(self.signup_inner_frame, show="*")
        self.confirm_password_entry.grid(row=2, column=1, padx=(0,10), pady=5, sticky="ew")

        # Bind Enter key
        self.username_entry.bind("<Return>", lambda event: signup())
        self.password_entry.bind("<Return>", lambda event: signup())
        self.confirm_password_entry.bind("<Return>", lambda event: signup())

        # Define signup function
        def signup():
            username = self.username_entry.get()
            password = self.password_entry.get()
            confirm_password = self.confirm_password_entry.get()

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
                # Clear entries
                self.username_entry.delete(0, "end")
                self.password_entry.delete(0, "end")
                self.confirm_password_entry.delete(0, "end")
                self.controller.show_page("LoginPage")

        # Buttons
        self.signup_button = ctk.CTkButton(
            self.signup_inner_frame,
            text="Sign Up",
            command=signup,
            width=200,
            height=40,
            fg_color="#111111",
            hover_color="#333333",
            text_color="#FFFFFF",
            corner_radius=8
        )
        self.signup_button.grid(row=3, column=0, columnspan=2, pady=(20,10))

        self.back_button = ctk.CTkButton(
            self.signup_inner_frame,
            text="Back to Login",
            width=200,
            height=40,
            command=lambda: self.controller.show_page("LoginPage"),
            fg_color="#3B8ED0",
            hover_color="#1C6EA4",
            text_color="#FFFFFF",
            corner_radius=8
        )
        self.back_button.grid(row=4, column=0, columnspan=2, pady=5)
