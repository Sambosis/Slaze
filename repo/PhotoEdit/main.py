from typing import Callable
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageOps
from image_processor import ImageProcessor

class PhotoEditor:
    def __init__(self):
        self.root = Tk()
        self.root.title("Photo Editor")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        self.setup_ui()
        self.root.mainloop()
    def setup_ui(self):
        # Top toolbar
        toolbar = Frame(self.root, bg="lightgray", relief=RAISED, bd=2)
        toolbar.pack(side=TOP, fill=X, pady=5)

        # Load/Save group
        fs1 = Frame(toolbar)
        fs1.pack(side=LEFT, padx=5)
        Button(fs1, text="Open", command=self.load_dialog, width=8).pack(side=LEFT, padx=2)
        Button(fs1, text="Save", command=self.save_dialog, width=8).pack(side=LEFT, padx=2)

        # Undo/Redo group
        fs2 = Frame(toolbar)
        fs2.pack(side=LEFT, padx=5)
        Button(fs2, text="Undo", command=self.undo, width=8).pack(side=LEFT, padx=2)
        Button(fs2, text="Redo", command=self.redo, width=8).pack(side=LEFT, padx=2)

        # Rotate/Flip group
        fs3 = Frame(toolbar)
        fs3.pack(side=LEFT, padx=5)
        Button(fs3, text="Rotate CW", command=lambda: self.apply_filter(ImageProcessor.rotate_cw), width=10).pack(side=LEFT, padx=1)
        Button(fs3, text="Rotate CCW", command=lambda: self.apply_filter(ImageProcessor.rotate_ccw), width=10).pack(side=LEFT, padx=1)
        Button(fs3, text="Flip H", command=lambda: self.apply_filter(ImageProcessor.flip_horizontal), width=8).pack(side=LEFT, padx=1)
        Button(fs3, text="Flip V", command=lambda: self.apply_filter(ImageProcessor.flip_vertical), width=8).pack(side=LEFT, padx=1)

        # Edit group
        fs4 = Frame(toolbar)
        fs4.pack(side=LEFT, padx=5)
        Button(fs4, text="Apply Crop", command=self.apply_crop, width=10).pack(side=LEFT, padx=2)
        Button(fs4, text="Resize...", command=self.resize_dialog, width=10).pack(side=LEFT, padx=2)

        # Filters group
        fs5 = Frame(toolbar)
        fs5.pack(side=LEFT, padx=5)
        Button(fs5, text="Grayscale", command=lambda: self.apply_filter(ImageProcessor.apply_grayscale), width=10).pack(side=LEFT, padx=1)
        Button(fs5, text="Sepia", command=lambda: self.apply_filter(ImageProcessor.apply_sepia), width=10).pack(side=LEFT, padx=1)
        Button(fs5, text="Blur", command=lambda: self.apply_filter(ImageProcessor.gaussian_blur), width=8).pack(side=LEFT, padx=1)
        Button(fs5, text="Sharpen", command=lambda: self.apply_filter(ImageProcessor.sharpen), width=8).pack(side=LEFT, padx=1)

        # Adjustments group
        fs6 = Frame(toolbar)
        fs6.pack(side=LEFT, padx=5)
        Button(fs6, text="Brightness...", command=self.brightness_dialog, width=12).pack(side=LEFT, padx=2)
        Button(fs6, text="Contrast...", command=self.contrast_dialog, width=12).pack(side=LEFT, padx=2)

        # Central scrollable canvas
        self.scroll_frame = Frame(self.root)
        self.scroll_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)

        self.canvas = Canvas(self.scroll_frame, bg='white', highlightthickness=0)
        self.vbar = Scrollbar(self.scroll_frame, orient=VERTICAL, command=self.canvas.yview)
        self.hbar = Scrollbar(self.scroll_frame, orient=HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)

        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.vbar.pack(side=RIGHT, fill=Y)
        self.hbar.pack(side=BOTTOM, fill=X)

        # Event bindings
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-1>", self.on_crop_start)
        self.canvas.bind("<B1-Motion>", self.on_crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_crop_end)
        self.canvas.bind("<Configure>", self.on_configure)

        # Bottom status bar
        status_frame = Frame(self.root, relief=SUNKEN, bd=2)
        status_frame.pack(side=BOTTOM, fill=X)
        self.status_label = Label(status_frame, text="No image loaded", anchor="w", padx=5)
        self.status_label.pack(fill=X)

        # Initialize variables
        self.image_history = []
        self.history_index = -1
        self.current_image = None
        self.zoom_factor = 1.0
        self.crop_start = None
        self.photo = None
        self.image_label = None
    def load_dialog(self):
        filepath = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
        )
        if filepath and self.load_image(filepath):
            pass
    def load_image(self, filepath: str) -> bool:
        try:
            img = Image.open(filepath).convert('RGB')
            self.image_history = [img.copy()]
            self.history_index = 0
            self.current_image = img.copy()
            self.zoom_factor = 1.0
            self.canvas.delete("crop_rect")
            self.update_canvas()
            self.update_status()
            return True
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load image:\n{str(e)}")
            return False

    def save_dialog(self):
        if not self.current_image:
            messagebox.showwarning("Save", "No image loaded.")
            return
        filepath = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")]
        )
        if not filepath:
            return
        if filepath.lower().endswith(('.png', '.PNG')):
            self.current_image.save(filepath, "PNG")
            messagebox.showinfo("Saved", "Image saved as PNG.")
        else:
            self._jpeg_quality_dialog(filepath)

    def _jpeg_quality_dialog(self, filepath):
        top = Toplevel(self.root)
        top.title("JPEG Quality")
        top.geometry("280x120")
        top.transient(self.root)
        top.grab_set()
        Label(top, text="Quality (1-100):").pack(pady=10)
        quality_var = IntVar(value=90)
        scale = Scale(top, from_=1, to=100, orient=HORIZONTAL, variable=quality_var, length=200)
        scale.pack(pady=5)
        Label(top, textvariable=quality_var).pack()
        def save():
            try:
                self.current_image.save(filepath, "JPEG", quality=quality_var.get(), optimize=True)
                messagebox.showinfo("Saved", "Image saved as JPEG.")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))
            top.destroy()
        Button(top, text="Save", command=save, default=ACTIVE).pack(pady=15)
        top.bind("<Return>", lambda e: save())

    def apply_filter(self, filter_func):
        if not self.current_image:
            return
        try:
            new_img = filter_func(self.current_image.copy())
            self.add_to_history(new_img)
            self.canvas.delete("crop_rect")
            self.update_canvas()
            self.update_status()
        except Exception as e:
            messagebox.showerror("Apply Error", str(e))

    def add_to_history(self, new_img):
        self.image_history = self.image_history[:self.history_index + 1] + [new_img]
        self.history_index = len(self.image_history) - 1
        if len(self.image_history) > 10:
            self.image_history.pop(0)
            self.history_index -= 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_image = self.image_history[self.history_index].copy()
            self.canvas.delete("crop_rect")
            self.update_canvas()
            self.update_status()

    def redo(self):
        if self.history_index < len(self.image_history) - 1:
            self.history_index += 1
            self.current_image = self.image_history[self.history_index].copy()
            self.canvas.delete("crop_rect")
            self.update_canvas()
            self.update_status()

    def update_canvas(self):
        if self.current_image is None:
            return
        w, h = self.current_image.size
        disp_w = int(w * self.zoom_factor)
        disp_h = int(h * self.zoom_factor)
        disp_img = self.current_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(disp_img)
        self.canvas.delete("image")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw", tags="image")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_status(self):
        if self.current_image:
            w, h = self.current_image.size
            self.status_label.config(text=f"{w}x{h}  Zoom: {int(self.zoom_factor * 100)}%")
        else:
            self.status_label.config(text="No image loaded")

    def on_zoom(self, event):
        old_zf = self.zoom_factor
        zoom_step = 1.25 if event.delta > 0 else 1 / 1.25
        new_zf = max(0.25, min(4.0, old_zf * zoom_step))
        zoom_scale = new_zf / old_zf
        old_cx = self.canvas.canvasx(event.x)
        old_cy = self.canvas.canvasy(event.y)
        self.zoom_factor = new_zf
        self.update_canvas()
        vis_w = self.canvas.winfo_width()
        vis_h = self.canvas.winfo_height()
        if vis_w == 1 or vis_h == 1:  # Not yet sized
            return
        sr = self.canvas.cget("scrollregion").split()
        total_w = float(sr[2])
        total_h = float(sr[3])
        new_cx = old_cx * zoom_scale
        new_cy = old_cy * zoom_scale
        frac_x = max(0.0, min(1.0, (new_cx - event.x * (total_w / vis_w)) / total_w))
        frac_y = max(0.0, min(1.0, (new_cy - event.y * (total_h / vis_h)) / total_h))
        self.canvas.xview_moveto(frac_x)
        self.canvas.yview_moveto(frac_y)

    def on_crop_start(self, event):
        self.crop_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.canvas.delete("crop_rect")
        self.canvas.create_rectangle(0, 0, 0, 0, outline="red", width=2, dash=(5, 5), tags="crop_rect")

    def on_crop_drag(self, event):
        if self.crop_start:
            cx1, cy1 = self.crop_start
            cx2 = self.canvas.canvasx(event.x)
            cy2 = self.canvas.canvasy(event.y)
            self.canvas.coords("crop_rect", cx1, cy1, cx2, cy2)

    def on_crop_end(self, event):
        self.crop_start = None

    def apply_crop(self):
        bbox = self.canvas.bbox("crop_rect")
        if bbox:
            x1, y1, x2, y2 = [coord / self.zoom_factor for coord in bbox]
            if x2 > x1 and y2 > y1:
                box = (int(x1), int(y1), int(x2), int(y2))
                try:
                    new_img = ImageProcessor.crop_image(self.current_image, box)
                    self.add_to_history(new_img)
                    self.canvas.delete("crop_rect")
                    self.update_canvas()
                    self.update_status()
                except Exception as e:
                    messagebox.showerror("Crop Error", str(e))

    def resize_dialog(self):
        if not self.current_image:
            return
        orig_w, orig_h = self.current_image.size
        ratio = orig_h / orig_w if orig_w > 0 else 1.0
        top = Toplevel(self.root)
        top.title("Resize Image")
        top.geometry("320x220")
        top.transient(self.root)
        top.grab_set()
        aspect_var = BooleanVar(value=True)
        w_var = IntVar(value=orig_w)
        h_var = IntVar(value=orig_h)

        def on_width_change(*args):
            if aspect_var.get():
                h_var.set(round(w_var.get() * ratio))

        def on_height_change(*args):
            if aspect_var.get() and ratio > 0:
                w_var.set(round(h_var.get() / ratio))

        Checkbutton(top, text="Maintain aspect ratio", variable=aspect_var).pack(pady=10)

        # Width
        row_w = Frame(top)
        row_w.pack(pady=5, padx=20, fill=X)
        Label(row_w, text="Width:").pack(side=LEFT)
        Scale(row_w, from_=1, to=10000, orient=HORIZONTAL, variable=w_var, length=150, command=lambda v: on_width_change()).pack(side=LEFT, padx=5)
        Label(row_w, textvariable=w_var, width=6).pack(side=RIGHT)

        # Height
        row_h = Frame(top)
        row_h.pack(pady=5, padx=20, fill=X)
        Label(row_h, text="Height:").pack(side=LEFT)
        Scale(row_h, from_=1, to=10000, orient=HORIZONTAL, variable=h_var, length=150, command=lambda v: on_height_change()).pack(side=LEFT, padx=5)
        Label(row_h, textvariable=h_var, width=6).pack(side=RIGHT)

        w_var.trace_add('write', on_width_change)
        h_var.trace_add('write', on_height_change)

        def apply_resize():
            new_w, new_h = w_var.get(), h_var.get()
            if new_w > 0 and new_h > 0:
                try:
                    new_img = ImageProcessor.resize_image(self.current_image, (new_w, new_h))
                    self.add_to_history(new_img)
                    self.update_canvas()
                    self.update_status()
                except Exception as e:
                    messagebox.showerror("Resize Error", str(e))
            top.destroy()

        Button(top, text="Apply", command=apply_resize, bg="lightgreen", width=10).pack(pady=20)
    def brightness_dialog(self):
        if not self.current_image:
            return
        top = Toplevel(self.root)
        top.title("Adjust Brightness")
        top.geometry("280x140")
        top.transient(self.root)
        top.grab_set()
        Label(top, text="Brightness (-100 to +100):").pack(pady=10)
        bright_var = IntVar(value=0)
        scale = Scale(top, from_=-100, to=100, orient=HORIZONTAL, variable=bright_var, length=250)
        scale.pack(pady=5)
        Label(top, textvariable=bright_var).pack()
        def apply():
            new_img = ImageProcessor.adjust_brightness(self.current_image.copy(), bright_var.get())
            self.add_to_history(new_img)
            self.update_canvas()
            self.update_status()
            top.destroy()
        Button(top, text="Apply", command=apply).pack(pady=15)
    def contrast_dialog(self):
        if not self.current_image:
            return
        top = Toplevel(self.root)
        top.title("Adjust Contrast")
        top.geometry("280x140")
        top.transient(self.root)
        top.grab_set()
        Label(top, text="Contrast (-100 to +100):").pack(pady=10)
        cont_var = IntVar(value=0)
        scale = Scale(top, from_=-100, to=100, orient=HORIZONTAL, variable=cont_var, length=250)
        scale.pack(pady=5)
        Label(top, textvariable=cont_var).pack()
        def apply():
            new_img = ImageProcessor.adjust_contrast(self.current_image.copy(), cont_var.get())
            self.add_to_history(new_img)
            self.update_canvas()
            self.update_status()
            top.destroy()
        Button(top, text="Apply", command=apply).pack(pady=15)

    def on_configure(self, event):
        self.update_status()

if __name__ == "__main__":
    PhotoEditor()
