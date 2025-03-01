import customtkinter as ctk


class DualScrollableFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs,  width=800, height=600)
        self.grid_propagate(False)
        # Create a canvas and two scrollbars
        self.canvas = ctk.CTkCanvas(self, bg=self.cget("fg_color"))
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.v_scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")

        self.h_scrollbar = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        # Create an inner frame to hold the content
        self.inner_frame = ctk.CTkFrame(self.canvas, fg_color=self.cget("fg_color"))
        #self.inner_frame.pack(expand=True, fill="both")
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        # Update scrollregion when the inner frame changes size
        self.inner_frame.bind("<Configure>", self._on_configure)

        # Bind mouse wheel scrolling on the canvas
        self.canvas.bind("<Enter>", self._bind_to_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_from_mousewheel)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _on_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _bind_to_mousewheel(self, event):
        # For Windows and MacOS
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # For Linux, you might also need:
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_from_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        # Windows / MacOS: event.delta is in multiples of 120 (or -120)
        if event.num == 4:  # Linux scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Linux scroll down
            self.canvas.yview_scroll(1, "units")
        else:
            # Windows and MacOS
            self.canvas.yview_scroll(-int(event.delta / 120), "units")
