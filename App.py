import tkinter as tk
from tkinter import messagebox
import json
import os
import bcrypt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Eyeliner Design App")
        self.geometry("800x600")

        # Container for all pages
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        # Dictionary to store pages
        self.pages = {}

        # Initialize pages
        self.show_page(LoginPage)

    def show_page(self, page_class):
        """Show a page by class."""
        if page_class not in self.pages:
            # Create the page if it doesn't exist
            page = page_class(parent=self.container, controller=self)
            self.pages[page_class] = page
            page.grid(row=0, column=0, sticky="nsew")
        # Bring the page to the front
        self.pages[page_class].tkraise()

class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()
        self.users = load_users()

    def create_widgets(self):
        tk.Label(self, text="Login", font=("Arial", 24)).pack(pady=20)
        tk.Label(self, text="Username").pack()
        username_entry = tk.Entry(self)
        username_entry.pack()
        tk.Label(self, text="Password").pack()
        password_entry = tk.Entry(self, show="*")
        password_entry.pack()

        def login():
            username = username_entry.get()
            password = password_entry.get()
            if username in self.users and check_password(self.users[username], password):
                self.controller.show_page(HomePage)
            else:
                messagebox.showerror("Login Failed", "Invalid username or password")


        def go_to_signup():
            self.controller.show_page(SignUpPage)

        tk.Button(self, text="Login", command=login).pack(pady=10)
        tk.Button(self, text="Sign Up", command=go_to_signup).pack(pady=5)

from Trial import *

class DesignPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.selected_designs = [False] * 6  # Keep track of selected designs
        self.create_widgets()
        self.designing = False

    def create_widgets(self):
        # Instructions
        while(self.designing==False):
            tk.Button(self, text="Start designing", command=lambda: start_designing).pack(pady=10)
        def start_designing():
            self.designing = True

        while(self.designing):
            tk.Label(self, text="Pick your favorite designs", font=("Arial", 18)).pack(pady=20)
            # Frame for the design canvas
            canvas_frame = tk.Frame(self)
            canvas_frame.pack(fill="both", expand=True)

            # Generate the design grid
            self.fig, self.axes = initialise_gene_pool()
            self.fig.tight_layout(pad=3)

            # Fill the subplots with dummy data
            self.buttons = []
            for i, ax in enumerate(self.axes.flatten()):
                x = np.linspace(0, 10, 100)
                y = np.sin(x + i)  # Generate different designs
                ax.plot(x, y)
                ax.set_title(f"Design {i + 1}")

                # Add button functionality to each subplot
                button = tk.Button(
                    canvas_frame,
                    text="Select",
                    bg="red",
                    fg="white",
                    command=lambda idx=i: self.toggle_selection(idx),
                )
                self.buttons.append(button)

            # Draw the designs on the canvas
            canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()

            # Place toggle buttons below each subplot
            for i, button in enumerate(self.buttons):
                row, col = divmod(i, 3)  # Convert index into 2D grid positions
                button.place(
                    x=col * 200 + 50, y=row * 200 + 350, width=100, height=40
                )  # Adjust positions as needed

            # Add a submission button
            tk.Button(self, text="Submit Selection", command=self.submit_selection, bg="blue", fg="white").pack(pady=10)
        tk.Button(self, text="Back to Home", command=lambda: self.controller.show_page(HomePage)).pack(pady=10)

    def toggle_selection(self, idx):
        """Toggle the selection state of a design."""
        self.selected_designs[idx] = not self.selected_designs[idx]
        # Update button appearance
        self.buttons[idx].config(bg="green" if self.selected_designs[idx] else "red", text="Selected" if self.selected_designs[idx] else "Select")

    def submit_selection(self):
        """Process and validate the selected designs."""
        selected = [i for i, selected in enumerate(self.selected_designs) if selected]
        if len(selected) != 2:
            tk.messagebox.showerror("Error", "Please select exactly 2 designs.")
        else:
            tk.messagebox.showinfo("Selection Complete", f"You selected designs: {selected}")
            # Continue with merging or further processing

class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Welcome to the Eyeliner Design App", font=("Arial", 18)).pack(pady=20)
        tk.Button(self, text="Start Designing", command=lambda: self.controller.show_page(DesignPage)).pack(pady=10)


class SignUpPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.users = load_users()
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Sign Up", font=("Arial", 24)).pack(pady=20)

        tk.Label(self, text="Username").pack()
        username_entry = tk.Entry(self)
        username_entry.pack()

        tk.Label(self, text="Password").pack()
        password_entry = tk.Entry(self, show="*")
        password_entry.pack()

        tk.Label(self, text="Confirm Password").pack()
        confirm_password_entry = tk.Entry(self, show="*")
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
                self.users[username] = hash_password(password)
                save_users(self.users)
                messagebox.showinfo("Success", "Account created successfully")
                self.controller.show_page(LoginPage)


        tk.Button(self, text="Sign Up", command=signup).pack(pady=10)
        tk.Button(self, text="Back to Login", command=lambda: self.controller.show_page(LoginPage)).pack(pady=5)

App().mainloop()

"""
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
        #Destroy the current frame and replace it with a new one.
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

        self.apply_style(frame, style = "pastel")
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

        self.apply_style(frame, style = "pastel")
        self.switch_frame(frame)

    def show_home_page(self):
        frame = tk.Frame(self.root)

        tk.Label(frame, text="Welcome to the Eyeliner Design App", font=("Arial", 18)).pack(pady=20)
        tk.Button(frame, text="Start Designing", command=self.show_design_page).pack(pady=10)

        self.apply_style(frame, style = "pastel")
        self.switch_frame(frame)

    def show_design_page(self):
        frame = tk.Frame(self.root)

        tk.Label(frame, text="Design Your Eyeliner", font=("Arial", 18)).pack(pady=20)

        def run_design():
            fig = initialise_gene_pool()  # Generate the designs
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()
            tk.Label(frame, text="Pick your 2 favorite designs", font=("Arial", 18)).pack(pady=20)

        tk.Button(frame, text="Start", command=run_design).pack(pady=10)
        tk.Button(frame, text="Back", command=self.show_home_page).pack(pady=10)

        self.apply_style(frame, style = "pastel")
        self.switch_frame(frame)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
"""