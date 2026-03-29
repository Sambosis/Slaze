import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable, Dict, List, Optional, Sequence


class DrumMachineUI:
    TRACK_NAMES: Sequence[str] = ("Kick", "Snare", "Hi-Hat", "Clap")
    STEP_COUNT: int = 16

    COLOR_ACTIVE = "#4CAF50"
    COLOR_INACTIVE = "#1C1C1C"
    COLOR_CURSOR_ACTIVE = "#FBC02D"
    COLOR_CURSOR_INACTIVE = "#6D4C41"
    COLOR_TEXT = "#FFFFFF"
    COLOR_BG = "#0E0E0E"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Python Drum Machine")
        self.root.configure(bg=self.COLOR_BG)

        self.frames: Dict[str, ttk.Frame] = {}
        self.step_buttons: List[List[tk.Button]] = [[] for _ in self.TRACK_NAMES]
        self.step_states: List[List[bool]] = [
            [False] * self.STEP_COUNT for _ in self.TRACK_NAMES
        ]

        self.sample_selectors: List[ttk.Combobox] = []
        self.sample_selector_vars: List[tk.StringVar] = []

        self.tempo_slider: Optional[tk.Scale] = None
        self.swing_slider: Optional[tk.Scale] = None
        self.volume_slider: Optional[tk.Scale] = None
        self.play_button: Optional[ttk.Button] = None
        self.stop_button: Optional[ttk.Button] = None

        self.status_var = tk.StringVar(value="Ready.")
        self.status_label: Optional[ttk.Label] = None

        self.preset_name_var = tk.StringVar(value="preset_01")
        self.preset_entry: Optional[ttk.Entry] = None

        self._cursor_step: Optional[int] = None

        self.on_toggle_step: Optional[Callable[[int, int], None]] = None
        self.on_play: Optional[Callable[[], None]] = None
        self.on_stop: Optional[Callable[[], None]] = None
        self.on_tempo_change: Optional[Callable[[int], None]] = None
        self.on_swing_change: Optional[Callable[[float], None]] = None
        self.on_volume_change: Optional[Callable[[float], None]] = None
        self.on_sample_change: Optional[Callable[[int, str], None]] = None
        self.on_sample_browse: Optional[Callable[[int, str], None]] = None
        self.on_save_preset: Optional[Callable[[str], None]] = None
        self.on_load_preset: Optional[Callable[[str], None]] = None

        self._build_layout()

    def bind_callbacks(
        self,
        *,
        on_toggle_step: Callable[[int, int], None],
        on_play: Callable[[], None],
        on_stop: Callable[[], None],
        on_tempo_change: Callable[[int], None],
        on_swing_change: Callable[[float], None],
        on_volume_change: Callable[[float], None],
        on_sample_change: Callable[[int, str], None],
        on_sample_browse: Callable[[int, str], None],
        on_save_preset: Callable[[str], None],
        on_load_preset: Callable[[str], None],
    ) -> None:
        self.on_toggle_step = on_toggle_step
        self.on_play = on_play
        self.on_stop = on_stop
        self.on_tempo_change = on_tempo_change
        self.on_swing_change = on_swing_change
        self.on_volume_change = on_volume_change
        self.on_sample_change = on_sample_change
        self.on_sample_browse = on_sample_browse
        self.on_save_preset = on_save_preset
        self.on_load_preset = on_load_preset

    def refresh_grid(self, pattern: Sequence[Sequence[bool]]) -> None:
        def apply() -> None:
            for track_idx in range(len(self.step_states)):
                row = pattern[track_idx] if track_idx < len(pattern) else []
                for step_idx in range(self.STEP_COUNT):
                    state = bool(row[step_idx]) if step_idx < len(row) else False
                    self.step_states[track_idx][step_idx] = state
                    self._apply_step_style(track_idx, step_idx)
            self._cursor_step = None

        self._safe_call(apply)

    def update_step_button(self, track_index: int, step_index: int, state: bool) -> None:
        def apply() -> None:
            if not self._valid_indices(track_index, step_index):
                return
            self.step_states[track_index][step_index] = state
            self._apply_step_style(track_index, step_index)

        self._safe_call(apply)

    def update_step_cursor(self, step_index: int) -> None:
        def apply() -> None:
            if self.STEP_COUNT == 0:
                return
            self._cursor_step = step_index % self.STEP_COUNT
            for track in range(len(self.step_states)):
                for step in range(self.STEP_COUNT):
                    self._apply_step_style(track, step)

        self._safe_call(apply)

    def set_transport_state(self, is_playing: bool) -> None:
        def apply() -> None:
            if self.play_button:
                self.play_button.config(state="disabled" if is_playing else "normal")
            if self.stop_button:
                self.stop_button.config(state="normal" if is_playing else "disabled")

        self._safe_call(apply)

    def populate_sample_options(self, options: Sequence[str]) -> None:
        values = tuple(options)

        def apply() -> None:
            for selector in self.sample_selectors:
                selector.config(values=values)

        self._safe_call(apply)

    def set_sample_selection(self, track_index: int, value: str) -> None:
        def apply() -> None:
            if 0 <= track_index < len(self.sample_selector_vars):
                self.sample_selector_vars[track_index].set(value)

        self._safe_call(apply)

    def set_tempo(self, bpm: int) -> None:
        def apply() -> None:
            if self.tempo_slider:
                self.tempo_slider.set(max(60, min(200, bpm)))

        self._safe_call(apply)

    def set_swing(self, swing_percent: float) -> None:
        def apply() -> None:
            if self.swing_slider:
                self.swing_slider.set(max(0.0, min(60.0, swing_percent)))

        self._safe_call(apply)

    def set_volume(self, volume: float) -> None:
        def apply() -> None:
            if self.volume_slider:
                self.volume_slider.set(max(0.0, min(1.0, volume)) * 100.0)

        self._safe_call(apply)

    def get_preset_name(self) -> str:
        return self.preset_name_var.get().strip()

    def set_preset_name(self, name: str) -> None:
        def apply() -> None:
            self.preset_name_var.set(name)

        self._safe_call(apply)

    def set_status_message(self, message: str) -> None:
        def apply() -> None:
            self.status_var.set(message)

        self._safe_call(apply)

    def _build_layout(self) -> None:
        self._configure_style()

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=16)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        for row in range(5):
            main.rowconfigure(row, weight=0)
        main.rowconfigure(3, weight=1)

        ttk.Label(main, text="Tk Drum Machine", font=("Segoe UI", 20, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )

        self._build_controls_frame(main, row=1)
        self._build_transport_frame(main, row=2)
        self._build_step_grid(main, row=3)
        self._build_presets_frame(main, row=4)
        self._build_status_bar()

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=self.COLOR_BG)
        style.configure("TLabel", background=self.COLOR_BG, foreground=self.COLOR_TEXT)
        style.configure("TLabelframe", background=self.COLOR_BG, foreground=self.COLOR_TEXT)
        style.configure("TLabelframe.Label", background=self.COLOR_BG, foreground=self.COLOR_TEXT)
        style.configure("Primary.TButton", background=self.COLOR_ACTIVE, foreground=self.COLOR_TEXT)
        style.map("Primary.TButton", background=[("active", "#66BB6A")])
        style.configure("Status.TLabel", background="#1C1C1C", foreground="#E0E0E0")

    def _build_controls_frame(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.LabelFrame(parent, text="Global Controls", padding=12)
        frame.grid(row=row, column=0, sticky="ew", pady=(12, 8))
        frame.columnconfigure((0, 1, 2), weight=1, uniform="controls")

        self.tempo_slider = self._create_slider(
            frame, column=0, label="Tempo (BPM)", from_=60, to=200, command=self._handle_tempo_slider
        )
        self.swing_slider = self._create_slider(
            frame, column=1, label="Swing (%)", from_=0, to=60, command=self._handle_swing_slider
        )
        self.volume_slider = self._create_slider(
            frame, column=2, label="Master Volume", from_=0, to=100, command=self._handle_volume_slider
        )

        self.tempo_slider.set(120)

    def _create_slider(
        self,
        parent: ttk.Frame,
        *,
        column: int,
        label: str,
        from_: float,
        to: float,
        command: Callable[[str], None],
    ) -> tk.Scale:
        ttk.Label(parent, text=label).grid(row=0, column=column, sticky="ew", padx=4)
        slider = tk.Scale(
            parent,
            from_=from_,
            to=to,
            orient="horizontal",
            command=command,
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT,
            highlightbackground=self.COLOR_BG,
            troughcolor=self.COLOR_INACTIVE,
        )
        slider.grid(row=1, column=column, sticky="ew", padx=4)
        return slider

    def _build_transport_frame(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=8)
        frame.columnconfigure((0, 1), weight=0)

        self.play_button = self._create_button(
            frame, text="▶ Play", command=lambda: self.on_play and self.on_play(), style="Primary.TButton"
        )
        self.play_button.grid(row=0, column=0, padx=(0, 4))

        self.stop_button = self._create_button(
            frame, text="■ Stop", command=lambda: self.on_stop and self.on_stop(), state="disabled"
        )
        self.stop_button.grid(row=0, column=1)

    def _build_step_grid(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="nsew", pady=(8, 12))
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)

        for track_idx, name in enumerate(self.TRACK_NAMES):
            frame.rowconfigure(track_idx, weight=1)

            label_frame = ttk.Frame(frame)
            label_frame.grid(row=track_idx, column=0, sticky="ns", padx=(0, 12))
            label_frame.columnconfigure(0, weight=1)
            label_frame.rowconfigure(0, weight=0)
            label_frame.rowconfigure(1, weight=0)

            ttk.Label(label_frame, text=name, anchor="w").grid(row=0, column=0, sticky="ew")

            var = tk.StringVar()
            selector = ttk.Combobox(label_frame, textvariable=var, width=12)
            selector.grid(row=1, column=0, sticky="ew", pady=(4, 0))
            selector.bind(
                "<<ComboboxSelected>>",
                lambda e, i=track_idx: self.on_sample_change and self.on_sample_change(i, e.widget.get()),
            )
            self.sample_selectors.append(selector)
            self.sample_selector_vars.append(var)

        grid_frame = ttk.Frame(frame)
        grid_frame.grid(row=0, column=1, rowspan=len(self.TRACK_NAMES), sticky="nsew")

        for track_idx in range(len(self.TRACK_NAMES)):
            grid_frame.rowconfigure(track_idx, weight=1, uniform="row")
            self.step_buttons.append([])
            for step_idx in range(self.STEP_COUNT):
                grid_frame.columnconfigure(step_idx, weight=1, uniform="col")
                button = tk.Button(
                    grid_frame,
                    bg=self.COLOR_INACTIVE,
                    activebackground=self.COLOR_ACTIVE,
                    bd=1,
                    relief="raised",
                    command=lambda i=track_idx, j=step_idx: self._toggle_step_state(i, j),
                )
                button.grid(row=track_idx, column=step_idx, sticky="nsew", padx=1, pady=1)
                self.step_buttons[track_idx].append(button)

    def _build_presets_frame(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.LabelFrame(parent, text="Presets", padding=12)
        frame.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        frame.columnconfigure(0, weight=1)

        self.preset_entry = ttk.Entry(frame, textvariable=self.preset_name_var)
        self.preset_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=0, column=1)

        self._create_button(
            button_frame,
            text="Save",
            command=lambda: self.on_save_preset and self.on_save_preset(self.get_preset_name()),
        ).grid(row=0, column=0, padx=4)

        self._create_button(
            button_frame,
            text="Load",
            command=lambda: self.on_load_preset and self.on_load_preset(self.get_preset_name()),
        ).grid(row=0, column=1, padx=4)

    def _build_status_bar(self) -> None:
        status_bar = ttk.Frame(self.root, style="Status.TFrame", height=24)
        status_bar.grid(row=1, column=0, sticky="ew")
        self.status_label = ttk.Label(
            status_bar, textvariable=self.status_var, style="Status.TLabel", padding=(4, 2)
        )
        self.status_label.grid(row=0, column=0, sticky="ew")

    def _create_button(self, parent: ttk.Frame, **kwargs) -> ttk.Button:
        return ttk.Button(parent, **kwargs)

    def _toggle_step_state(self, track_index: int, step_index: int) -> None:
        if not self._valid_indices(track_index, step_index):
            return
        current_state = self.step_states[track_index][step_index]
        new_state = not current_state
        self.step_states[track_index][step_index] = new_state
        self._apply_step_style(track_index, step_index)
        if self.on_toggle_step:
            self.on_toggle_step(track_index, step_index)

    def _apply_step_style(self, track_index: int, step_index: int) -> None:
        if not self._valid_indices(track_index, step_index):
            return

        button = self.step_buttons[track_index][step_index]
        is_active = self.step_states[track_index][step_index]
        is_cursor = self._cursor_step == step_index

        if is_cursor:
            color = self.COLOR_CURSOR_ACTIVE if is_active else self.COLOR_CURSOR_INACTIVE
        else:
            color = self.COLOR_ACTIVE if is_active else self.COLOR_INACTIVE
        button.config(bg=color, activebackground=color)

    def _valid_indices(self, track_index: int, step_index: int) -> bool:
        return (
            0 <= track_index < len(self.step_states)
            and 0 <= step_index < len(self.step_states[track_index])
        )

    def _safe_call(self, callback: Callable[[], None]) -> None:
        try:
            self.root.after(0, callback)
        except tk.TclError:
            pass

    def _handle_tempo_slider(self, value: str) -> None:
        if self.on_tempo_change:
            self.on_tempo_change(int(value))

    def _handle_swing_slider(self, value: str) -> None:
        if self.on_swing_change:
            self.on_swing_change(float(value))

    def _handle_volume_slider(self, value: str) -> None:
        if self.on_volume_change:
            self.on_volume_change(float(value) / 100.0)