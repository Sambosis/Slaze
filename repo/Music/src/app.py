import tkinter as tk
from tkinter import messagebox, filedialog
from pathlib import Path
from typing import List, Optional
from itertools import repeat

from src.ui import DrumMachineUI
from src.sequencer import StepSequencer
from src.audio_engine import AudioEngine, load_default_samples
from src.presets import PresetManager, PresetData

# Constants for asset paths
ASSETS_DIR = Path(__file__).parent.parent / "assets" / "samples"
PRESETS_DIR = Path(__file__).parent.parent / "presets"

# Ensure directories exist
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
PRESETS_DIR.mkdir(parents=True, exist_ok=True)


class DrumMachineApp:
    """
    Main application class for the drum machine.
    
    Orchestrates the UI, sequencer, audio engine, and preset manager.
    Handles user interactions, default sample loading, playback controls,
    preset management, and graceful shutdown.
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the drum machine app.
        
        Args:
            root: The Tkinter root window.
        """
        self.root = root
        self.root.title("Drum Machine")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Components
        self.ui = DrumMachineUI(root)
        self.sequencer = StepSequencer()
        self.audio = AudioEngine()
        self.preset_manager = PresetManager(PRESETS_DIR)
        
        # Per-track sample file paths
        self.sample_paths: List[Optional[Path]] = [None] * 4
        
        # Bind UI callbacks
        self.ui.bind_callbacks(
            on_toggle_step=self.toggle_step,
            on_play=self.start_playback,
            on_stop=self.stop_playback,
            on_tempo_change=lambda v: self.update_tempo(str(v)),
            on_swing_change=lambda v: self.update_swing(str(v)),
            on_volume_change=lambda v: self.update_volume(str(v * 100)),  # Scale to 0-100
            on_sample_change=self.select_sample,
            on_sample_browse=self.select_sample,
            on_save_preset=self.save_preset,
            on_load_preset=self.load_preset,
        )
        
        # Load default samples
        default_samples = load_default_samples(ASSETS_DIR)
        track_names = ["Kick", "Snare", "Hi-Hat", "Clap"]
        
        # Populate UI with sample options
        sample_names = [p.name for p in default_samples.values()]
        self.ui.populate_sample_options(sample_names)

        for i, track_name in enumerate(track_names):
            sample_path = default_samples.get(track_name)
            if sample_path:
                self.sample_paths[i] = sample_path
                self.audio.load_sample(track_name, sample_path)
                self.ui.set_sample_selection(i, sample_path.name)
        
        # Set initial defaults: tempo 120, swing 0, volume 0.8, pattern off
        self.sequencer.tempo_bpm = 120
        self.sequencer.swing_ratio = 0.0
        self.audio.volume = 0.8
        self.ui.tempo_slider.set(120)
        self.ui.swing_slider.set(0)
        self.ui.volume_slider.set(80)  # 0.8 * 100
    
    def toggle_step(self, track_index: int, step_index: int) -> None:
        """
        Toggle the state of a step in the pattern and update UI.
        
        Args:
            track_index: Index of the track (0-3).
            step_index: Index of the step (0-15).
        """
        self.sequencer.pattern[track_index][step_index] = not self.sequencer.pattern[track_index][step_index]
        self.ui.update_step_button(
            track_index, step_index, self.sequencer.pattern[track_index][step_index]
        )
    
    def update_tempo(self, value: str) -> None:
        """
        Update the tempo based on slider value.
        
        Args:
            value: String representation of tempo (60-200).
        """
        try:
            tempo = int(float(value))
            if 60 <= tempo <= 200:
                self.sequencer.tempo_bpm = tempo
                if self.sequencer.is_running:
                    self.sequencer.stop()
                    self.sequencer.start(tempo, self.sequencer.swing_ratio, self.on_step_advance, self.on_note_trigger)
                self.ui.status_var.set("Tempo updated.")
            else:
                self.ui.status_var.set("Tempo must be between 60 and 200 BPM.")
        except ValueError:
            self.ui.status_var.set("Invalid tempo value.")
    
    def update_swing(self, value: str) -> None:
        """
        Update the swing ratio based on slider value.
        
        Args:
            value: String representation of swing (0-60).
        """
        try:
            swing = float(value)
            if 0 <= swing <= 60:
                self.sequencer.swing_ratio = swing / 100.0  # Convert to ratio
                if self.sequencer.is_running:
                    self.sequencer.stop()
                    self.sequencer.start(self.sequencer.tempo_bpm, self.sequencer.swing_ratio, self.on_step_advance, self.on_note_trigger)
                self.ui.status_var.set("Swing updated.")
            else:
                self.ui.status_var.set("Swing must be between 0 and 60%.")
        except ValueError:
            self.ui.status_var.set("Invalid swing value.")
    
    def update_volume(self, value: str) -> None:
        """
        Update the master volume based on slider value.
        
        Args:
            value: String representation of volume (0-100).
        """
        try:
            volume = float(value) / 100.0  # Normalize to 0-1
            if 0 <= volume <= 1:
                self.audio.volume = volume
                self.audio.set_master_volume(volume)
                self.ui.status_var.set("Volume updated.")
            else:
                self.ui.status_var.set("Volume must be between 0 and 100.")
        except ValueError:
            self.ui.status_var.set("Invalid volume value.")
    
    def select_sample(self, track_index: int, sample_name: str) -> None:
        """
        Handle sample selection for a track, opening file dialog or using combo.
        
        Args:
            track_index: Index of the track (0-3).
            sample_name: Selected sample name.
        """
        # If sample_name is not default, assume it's a file path or trigger dialog
        if sample_name in ["Kick", "Snare", "Hi-Hat", "Clap"] and self.sample_paths[track_index]:
            # Already loaded
            pass
        else:
            # Open file dialog for WAV
            filename = filedialog.askopenfilename(
                title="Select WAV Sample",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )
            if filename:
                path = Path(filename)
                if path.suffix.lower() == ".wav" and path.exists():
                    self.sample_paths[track_index] = path
                    track_name = ["Kick", "Snare", "Hi-Hat", "Clap"][track_index]
                    self.audio.load_sample(track_name, path)
                    self.ui.sample_selector_vars[track_index].set(path.name)  # Update UI to show filename
                    self.ui.status_var.set(f"Loaded sample: {path.name}")
                else:
                    messagebox.showerror("Error", "Invalid WAV file selected.")
            # Populate combo with available samples or just update
    
    def start_playback(self) -> None:
        """Start the sequencer playback."""
        if not self.sequencer.is_running:
            self.sequencer.start(
                self.sequencer.tempo_bpm,
                self.sequencer.swing_ratio,
                self.on_step_advance,
                self.on_note_trigger
            )
            self.ui.play_button.config(state="disabled")
            self.ui.stop_button.config(state="normal")
    
    def stop_playback(self) -> None:
        """Stop the sequencer playback."""
        if self.sequencer.is_running:
            self.sequencer.stop()
            self.ui.play_button.config(state="normal")
            self.ui.stop_button.config(state="disabled")
            self.ui.update_step_cursor(None)  # Clear cursor
    
    def on_step_advance(self, step_index: int) -> None:
        """
        Callback called when the sequencer advances to the next step.
        
        Args:
            step_index: The current step index.
        """
        # Use after() to update UI thread-safely
        self.root.after(0, self.ui.update_step_cursor, step_index)
    
    def on_note_trigger(self, step_index: int) -> None:
        """
        Callback called when notes are triggered at a step.
        
        Args:
            step_index: The step index where notes are played.
        """
        track_names = ["Kick", "Snare", "Hi-Hat", "Clap"]
        for track_index in range(4):
            if self.sequencer.pattern[track_index][step_index]:
                self.audio.play_sample(track_names[track_index], track_index)
    
    def save_preset(self, filename: Optional[str] = None) -> None:
        """
        Save the current configuration as a preset.
        
        Args:
            filename: Optional preset filename. If None, prompt user.
        """
        if filename is None:
            preset_name = self.ui.preset_name_var.get().strip()
            if not preset_name:
                preset_name = "preset_01"
            filename = str(PRESETS_DIR / f"{preset_name}.json")
        
        # Prepare preset data
        pattern_i = [[1 if b else 0 for b in row] for row in self.sequencer.pattern]
        samples = [str(path.name) if path else "" for path in self.sample_paths]
        data: PresetData = {
            "tempo": self.sequencer.tempo_bpm,
            "swing": int(self.sequencer.swing_ratio * 100),
            "volume": int(self.audio.volume * 100),
            "pattern": pattern_i,
            "samples": samples
        }
        
        try:
            self.preset_manager.save(filename, data)
            self.ui.status_var.set(f"Saved preset: {Path(filename).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset: {str(e)}")
    
    def load_preset(self, filename: Optional[str] = None) -> None:
        """
        Load a preset and update the application state.
        
        Args:
            filename: Optional preset filename. If None, prompt user.
        """
        if filename is None:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json")],
                initialdir=str(PRESETS_DIR)
            )
            if not filename:
                return
        
        try:
            data = self.preset_manager.load(filename)
            
            # Update sequencer
            self.sequencer.tempo_bpm = data["tempo"]
            self.sequencer.swing_ratio = data["swing"] / 100.0
            for row_idx, row in enumerate(data["pattern"]):
                self.sequencer.pattern[row_idx] = [b != 0 for b in row]
            
            # Update UI
            self.ui.tempo_slider.set(data["tempo"])
            self.ui.swing_slider.set(data["swing"])
            self.ui.volume_slider.set(data["volume"])
            for i, sample_name in enumerate(data["samples"]):
                # Assume sample_name is filename, find in presets or assets
                path = PRESETS_DIR / sample_name if sample_name else ASSETS_DIR / f"{['kick', 'snare', 'hihat', 'clap'][i]}.wav"
                self.sample_paths[i] = path if path.exists() else None
                self.ui.sample_selector_vars[i].set(sample_name or ["Kick", "Snare", "Hi-Hat", "Clap"][i])
            
            # Update audio
            self.audio.volume = data["volume"] / 100.0
            self.audio.set_master_volume(self.audio.volume)
            # Reload samples if needed
            track_names = ["Kick", "Snare", "Hi-Hat", "Clap"]
            for i in range(4):
                if self.sample_paths[i] and self.sample_paths[i].exists():
                    self.audio.load_sample(track_names[i], self.sample_paths[i])
            
            self.ui.refresh_grid()
            self.ui.status_var.set(f"Loaded preset: {Path(filename).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {str(e)}")
    
    def on_closing(self) -> None:
        """Handle graceful shutdown on window close."""
        self.stop_playback()
        self.audio.quit_mixer()
        self.root.destroy()