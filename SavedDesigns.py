import pickle
import customtkinter as ctk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from AnalyseDesign import analyse_positive, analyse_negative


class SaveDesignPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, width=1200, height=600, fg_color="#F9FAFB")
        self.grid_propagate(False)
        self.controller = controller

        # ========== TOP BAR WITH TITLE AND "BACK TO HOMEPAGE" BUTTON ==========
        self.top_bar_frame = ctk.CTkFrame(self, fg_color="#F9FAFB")
        self.top_bar_frame.pack(fill="x", pady=(10, 0))

        self.label_title = ctk.CTkLabel(
            self.top_bar_frame, text="Saved Designs",
            font=("Helvetica", 30, "bold"), text_color="#111111"
        )
        self.label_title.grid(row=0, column=0, sticky="w", padx=20)

        self.back_button = ctk.CTkButton(
            self.top_bar_frame, text="Back to Homepage",
            width=200, height=40, command=lambda: self.go_home(),
            fg_color="#3B8ED0", hover_color="#1C6EA4", font=("Helvetica", 14)
        )
        self.back_button.grid(row=0, column=1, sticky="e", padx=20)

        # ========== SCROLLABLE FRAME ==========
        self.outer_frame = ctk.CTkFrame(self, fg_color="#E8E8E8", corner_radius=15)
        self.outer_frame.pack(expand=True, fill="both", padx=20, pady=20)

        self.scrollable_frame = ctk.CTkScrollableFrame(
            self.outer_frame, fg_color="#FFFFFF", corner_radius=10
        )
        self.scrollable_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Enable fast mouse scrolling
        self.scrollable_frame.bind("<Enter>", self.bind_scroll_events)
        self.scrollable_frame.bind("<Leave>", self.unbind_scroll_events)

        # Call method to display designs
        self.update_designs()

    def bind_scroll_events(self, event):
        """ Bind mouse scrolling when cursor is inside the scrollable frame. """
        self.scrollable_frame.bind_all("<MouseWheel>", self._on_mouse_scroll)  # Windows & macOS
        self.scrollable_frame.bind_all("<Button-4>", self._on_mouse_scroll)  # Linux (scroll up)
        self.scrollable_frame.bind_all("<Button-5>", self._on_mouse_scroll)  # Linux (scroll down)

    def unbind_scroll_events(self, event):
        """ Unbind scrolling when cursor leaves the scrollable frame. """
        self.scrollable_frame.unbind_all("<MouseWheel>")
        self.scrollable_frame.unbind_all("<Button-4>")
        self.scrollable_frame.unbind_all("<Button-5>")

    def _on_mouse_scroll(self, event):
        """ Handle mouse scrolling for different OS with increased scroll speed. """
        SCROLL_STEP = 20  # Increase this for faster scrolling (default is 1)

        if event.num == 4:  # Linux Scroll Up
            self.scrollable_frame._parent_canvas.yview_scroll(-SCROLL_STEP, "units")
        elif event.num == 5:  # Linux Scroll Down
            self.scrollable_frame._parent_canvas.yview_scroll(SCROLL_STEP, "units")
        else:  # Windows/macOS (event.delta positive is up, negative is down)
            self.scrollable_frame._parent_canvas.yview_scroll(-SCROLL_STEP if event.delta > 0 else SCROLL_STEP, "units")

    def go_home(self):
        """ Go back to the homepage and update saved designs. """
        self.controller.pages["HomePage"].show_recent_designs()
        self.controller.show_page("HomePage")

    def update_designs(self):
        """ Clears and repopulates the scrollable frame with all saved designs. """
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        saved_designs = self.controller.all_saved_designs  # The global list of saved designs
        num_columns = 3  # Display designs in a 3-column grid

        for i, design in enumerate(saved_designs):
            design_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="#FFFFFF", corner_radius=10)
            design_frame.grid(row=i // num_columns, column=i % num_columns, padx=10, pady=10, sticky="nsew")

            fig = design.render_design(scale=1)

            for ax in fig.get_axes():
                ax.set_axis_off()

            #positive_score = analyse_positive(design, False)
            #negative_score = analyse_negative(design, False)
            #print(f"analyse_positive: {positive_score}")
            #print(f"negative Score: {negative_score}")

            canvas = FigureCanvasTkAgg(fig, master=design_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(expand=True, fill="both", padx=5, pady=5)
            plt.close(fig)

            canvas_widget.bind("<Button-1>", lambda event, d=design: self.show_design_popup(d))

            delete_button = ctk.CTkButton(
                design_frame, text="Delete",
                fg_color="#D9534F", hover_color="#C9302C",
                command=lambda d=design: self.confirm_delete(d),
                width=100
            )
            delete_button.pack(pady=(0, 5))

    def show_design_popup(self, design):
        """ Opens a popup showing a larger preview of the selected design. """
        popup = ctk.CTkToplevel(self)
        popup.title("Design Preview")
        popup.geometry("600x400")

        content_frame = ctk.CTkFrame(popup, fg_color="#F9FAFB")
        content_frame.pack(expand=True, fill="both", padx=10, pady=10)

        large_fig = design.render_design(scale=1.75)
        for ax in large_fig.get_axes():
            ax.set_axis_off()

        large_canvas = FigureCanvasTkAgg(large_fig, master=content_frame)
        large_canvas.draw()
        large_canvas.get_tk_widget().pack(expand=True, fill="both", padx=5, pady=5)

        close_button = ctk.CTkButton(content_frame, text="Close", command=popup.destroy)
        close_button.pack(pady=5)

    def confirm_delete(self, design):
        """ Opens a confirmation popup asking if the user wants to delete the design. """
        confirm_popup = ctk.CTkToplevel(self)
        confirm_popup.title("Confirm Delete")
        confirm_popup.geometry("300x150")

        frame = ctk.CTkFrame(confirm_popup, fg_color="#F9FAFB")
        frame.pack(expand=True, fill="both", padx=10, pady=10)

        label = ctk.CTkLabel(
            frame, text="Are you sure you want to delete this design?",
            font=("Helvetica", 14), text_color="#111111"
        )
        label.pack(pady=10)

        buttons_frame = ctk.CTkFrame(frame, fg_color="#F9FAFB")
        buttons_frame.pack()

        yes_button = ctk.CTkButton(
            buttons_frame, text="Yes",
            fg_color="#D9534F", hover_color="#C9302C",
            command=lambda d=design, p=confirm_popup: self.delete_design(d, p)
        )
        yes_button.grid(row=0, column=0, padx=10)

        no_button = ctk.CTkButton(
            buttons_frame, text="No",
            fg_color="#3B8ED0", hover_color="#1C6EA4",
            command=confirm_popup.destroy
        )
        no_button.grid(row=0, column=1, padx=10)

    def delete_design(self, design, popup):
        """ Remove the design from the saved list, update the pickle file, and refresh. """
        if design in self.controller.all_saved_designs:
            self.controller.all_saved_designs.remove(design)
            username = self.controller.current_user
            filename = f"{username}_designs.pkl"
            try:
                with open(filename, "wb") as f:
                    pickle.dump(self.controller.all_saved_designs, f)
            except Exception as e:
                print(f"Error updating saved designs file {filename}: {e}")
        popup.destroy()
        self.update_designs()
