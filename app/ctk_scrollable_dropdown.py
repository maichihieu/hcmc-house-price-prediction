'''
Code thư viện CTkScrollableDropdown
Nguồn: Akascape (Open Source MIT License)
'''
import customtkinter
import sys


class CTkScrollableDropdown(customtkinter.CTkToplevel):
    def __init__(self, attach, x=None, y=None, button_color=None, height=200, width=None,
                 fg_color=None, button_height=20, justify="center", scrollbar_button_color=None,
                 scrollbar=True, scrollbar_button_hover_color=None, frame_border_width=2,
                 values=[], command=None, image_values=[], alpha=0.97,
                 frame_corner_radius=20, double_click=False, resize=True, frame_border_color=None,
                 text_color=None, autocomplete=False, **button_kwargs):

        super().__init__(takefocus=1)

        self.focus()
        self.lift()
        self.alpha = alpha
        self.attach = attach
        self.corner = frame_corner_radius
        self.padding = 0
        self.focus_something = False
        self.disable = True

        if sys.platform.startswith("win"):
            self.after(100, lambda: self.overrideredirect(True))
            self.transparent_color = self._apply_appearance_mode(self._fg_color)
            self.attributes("-transparentcolor", self.transparent_color)
        elif sys.platform.startswith("darwin"):
            self.overrideredirect(True)
            self.attributes("-transparent", True)
        else:
            self.overrideredirect(True)
            self.wm_attributes("-type", "dropdown_menu")

        self.hide = True
        self.attach.bind('<Configure>', lambda e: self._withdraw() if not self.disable else None, add="+")
        self.attach.winfo_toplevel().bind('<Configure>', lambda e: self._withdraw() if not self.disable else None,
                                          add="+")
        self.attach.bind("<ButtonPress>", lambda e: self._withdraw() if not self.disable else None, add="+")
        self.bind("<Escape>", lambda e: self._withdraw() if not self.disable else None, add="+")
        self.bind("<FocusOut>", lambda e: self._withdraw() if not self.disable else None, add="+")

        if autocomplete:
            self.bind("<Button-1>", lambda e: self._withdraw() if not self.disable else None)
            self.attach.bind("<KeyRelease>", self._appear, add="+")
        else:
            self.attach.bind('<Button-1>', self._toggle_dropdown, add="+")

        self.attributes('-alpha', 0)
        self.disable = False
        self.fg_color = customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"] if fg_color is None else fg_color
        self.scroll_button_color = customtkinter.ThemeManager.theme["CTkScrollbar"][
            "button_color"] if scrollbar_button_color is None else scrollbar_button_color
        self.scroll_hover_color = customtkinter.ThemeManager.theme["CTkScrollbar"][
            "button_hover_color"] if scrollbar_button_hover_color is None else scrollbar_button_hover_color
        self.frame_border_color = customtkinter.ThemeManager.theme["CTkFrame"][
            "border_color"] if frame_border_color is None else frame_border_color
        self.button_color = customtkinter.ThemeManager.theme["CTkFrame"][
            "top_fg_color"] if button_color is None else button_color
        self.text_color = customtkinter.ThemeManager.theme["CTkLabel"][
            "text_color"] if text_color is None else text_color

        if scrollbar is False:
            self.scroll_button_color = self.fg_color
            self.scroll_hover_color = self.fg_color

        self.frame = customtkinter.CTkScrollableFrame(self, bg_color=self.transparent_color if sys.platform.startswith(
            "win") else None,
                                                      fg_color=self.fg_color,
                                                      scrollbar_button_hover_color=self.scroll_hover_color,
                                                      scrollbar_button_color=self.scroll_button_color,
                                                      border_width=frame_border_width, corner_radius=self.corner,
                                                      border_color=self.frame_border_color)
        self.frame._scrollbar.grid_configure(padx=3)
        self.frame.pack(expand=True, fill="both")

        self.dummy_entry = customtkinter.CTkEntry(self.frame, fg_color="transparent", border_width=0, height=1, width=1)
        self.no_match_label = customtkinter.CTkLabel(self.frame, text="No Match Found", text_color=self.text_color)

        self.values = values
        self.command = command
        self.image_values = image_values
        self.height = height
        self.width = width
        self.resize = resize
        self.autocomplete = autocomplete
        self.var_update = customtkinter.StringVar()
        self.appear = False

        if autocomplete:
            self.var_update.set(values[0])

        self.button_height = button_height
        self.justify = justify
        self.button_kwargs = button_kwargs
        self.double_click = double_click

    def _toggle_dropdown(self, event=None):
        if self.hide:
            self._appear(event)
        else:
            self._withdraw(event)

    def _withdraw(self, event=None):
        self.withdraw()
        self.hide = True

    def _appear(self, event=None):
        if self.autocomplete:
            x = self.attach.winfo_rootx()
            y = self.attach.winfo_rooty() + self.attach.winfo_height()

            if self.width is None:
                width = self.attach.winfo_width()
            else:
                width = self.width

            text = self.attach.get()
            self.frame.pack_forget()
            for widget in self.frame.winfo_children(): widget.destroy()

            new_values = [i for i in self.values if text.lower() in i.lower()]
            if not new_values:
                self.no_match_label.pack(fill="x", pady=5, padx=5)
            else:
                for i in new_values:
                    self._create_button(i)

            self.frame.pack(expand=True, fill="both")

        else:
            if self.appear: return
            x = self.attach.winfo_rootx()
            y = self.attach.winfo_rooty() + self.attach.winfo_height()

            if self.width is None:
                width = self.attach.winfo_width()
            else:
                width = self.width

            for widget in self.frame.winfo_children(): widget.destroy()
            for i in self.values:
                self._create_button(i)
            self.appear = True

        self.deiconify()
        self.geometry('{}x{}+{}+{}'.format(width, self.height, x, y))
        self.attributes('-alpha', self.alpha)
        self.hide = False

    def _create_button(self, value):
        button = customtkinter.CTkButton(self.frame, text=value, height=self.button_height,
                                         fg_color="transparent", text_color=self.text_color,
                                         anchor=self.justify, command=lambda v=value: self._command(v),
                                         **self.button_kwargs)
        if self.button_color:
            button.configure(hover_color=self.button_color)
        button.pack(fill="x", pady=2, padx=2)

    def _command(self, value):
        self.attach.delete(0, "end")
        self.attach.insert(0, value)
        if self.command: self.command(value)
        self._withdraw()