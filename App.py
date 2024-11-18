import tkinter as tk
from tkinter import messagebox
import json
import os
import bcrypt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Trial import *

# File to store user credentials
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

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Eyeliner Design App")
        self.root.geometry("800x600")
        self.current_frame = None
        self.users = load_users()

        # Initialize with login page
        self.show_login_page()

    def apply_style(self,frame, style):
        if style == "dark":
            self.apply_dark_mode_style(frame)
        elif style == "pastel":
            self.apply_pastel_style(frame)

        elif style == "vibrant":
            self.apply_vibrant_style(frame)

    def apply_dark_mode_style(self, frame):
        frame.configure(bg="#2c2c2c")
        for widget in frame.winfo_children():
            if isinstance(widget, tk.Label):
                widget.config(bg="#2c2c2c", font=("Roboto", 14), fg="white")
            elif isinstance(widget, tk.Button):
                widget.config(bg="#1e90ff", fg="white", font=("Roboto", 12), relief="flat", padx=10, pady=5)
                widget.config(activebackground="#1c86ee", activeforeground="white")
            elif isinstance(widget, tk.Entry):
                widget.config(bg="#444444", fg="white", font=("Roboto", 12), relief="solid", bd=1)

    def apply_pastel_style(self, frame):
        frame.configure(bg="#f3f4f9")
        for widget in frame.winfo_children():
            if isinstance(widget, tk.Label):
                widget.config(bg="#f3f4f9", font=("Segoe UI Rounded", 14), fg="#555555")
            elif isinstance(widget, tk.Button):
                widget.config(bg="#ffb6c1", fg="white", font=("Segoe UI Rounded", 12), relief="flat", padx=10, pady=5)
                widget.config(activebackground="#ff99aa", activeforeground="white")
            elif isinstance(widget, tk.Entry):
                widget.config(bg="#ffffff", fg="#555555", font=("Segoe UI Rounded", 12), relief="solid", bd=1)

    def apply_vibrant_style(self, frame):
        self.root.configure(bg="#f9f1c3")
        # Labels
        for widget in frame.winfo_children():
            if isinstance(widget, tk.Label):
                widget.config(bg="#f9f1c3", font=("Comic Sans MS", 14), fg="#2C3E50")

        # Buttons
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(bg="#ff6347", fg="white", font=("Comic Sans MS", 12), relief="flat", padx=10, pady=5)
                widget.config(activebackground="#ff4500", activeforeground="white")

        # Entry fields
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Entry):
                widget.config(bg="#fffacd", fg="#2C3E50", font=("Comic Sans MS", 12), relief="solid", bd=1)

    def switch_frame(self, new_frame):
        """Destroy the current frame and replace it with a new one."""
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = new_frame
        self.current_frame.pack(fill="both", expand=True)

    def show_login_page(self):
        frame = tk.Frame(self.root)

        tk.Label(frame, text="Login", font=("Arial", 24)).pack(pady=20)
        tk.Label(frame, text="Username").pack()
        username_entry = tk.Entry(frame)
        username_entry.pack()
        tk.Label(frame, text="Password").pack()
        password_entry = tk.Entry(frame, show="*")
        password_entry.pack()

        def login():
            username = username_entry.get()
            password = password_entry.get()
            if username in self.users and check_password(self.users[username], password):
                self.show_home_page()
            else:
                messagebox.showerror("Login Failed", "Invalid username or password")

        def go_to_signup():
            self.show_signup_page()

        tk.Button(frame, text="Login", command=login).pack(pady=10)
        tk.Button(frame, text="Sign Up", command=go_to_signup).pack(pady=5)

        self.apply_style(frame, style = "dark")
        self.switch_frame(frame)

    def show_signup_page(self):
        frame = tk.Frame(self.root)

        tk.Label(frame, text="Sign Up", font=("Arial", 24)).pack(pady=20)
        tk.Label(frame, text="Username").pack()
        username_entry = tk.Entry(frame)
        username_entry.pack()
        tk.Label(frame, text="Password").pack()
        password_entry = tk.Entry(frame, show="*")
        password_entry.pack()
        tk.Label(frame, text="Confirm Password").pack()
        confirm_password_entry = tk.Entry(frame, show="*")
        confirm_password_entry.pack()

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
                # Hash the password before saving
                self.users[username] = hash_password(password)
                save_users(self.users)
                messagebox.showinfo("Success", "Account created successfully")
                self.show_login_page()

        tk.Button(frame, text="Sign Up", command=signup).pack(pady=10)
        tk.Button(frame, text="Back to Login", command=self.show_login_page).pack(pady=5)

        self.apply_style(frame, style = "dark")
        self.switch_frame(frame)

    def show_home_page(self):
        frame = tk.Frame(self.root)

        tk.Label(frame, text="Welcome to the Eyeliner Design App", font=("Arial", 18)).pack(pady=20)
        tk.Button(frame, text="Start Designing", command=self.show_design_page).pack(pady=10)

        self.apply_style(frame, style = "dark")
        self.switch_frame(frame)

    def show_design_page(self):
        frame = tk.Frame(self.root)

        tk.Label(frame, text="Design Your Eyeliner", font=("Arial", 18)).pack(pady=20)

        def run_design():
            fig = initialise_gene_pool()  # Generate the designs
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()

        tk.Button(frame, text="Start", command=run_design).pack(pady=10)

        self.apply_style(frame, style = "dark")
        self.switch_frame(frame)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
