import threading
from time import sleep
from typing import Callable, List

class StepSequencer:
    """
    StepSequencer handles the pattern state and playback timing for the drum machine.
    It manages a 4x16 boolean grid of steps, tempo in BPM, swing ratio, and controls
    a background playback thread that advances through steps at calculated intervals,
    invoking callbacks for UI updates and audio triggering.
    
    Supports live editing of the pattern during playback by referencing shared state.
    Uses thread-safe controls with an Event for stopping and a Lock for pattern access.
    """
    
    TRACK_COUNT = 4
    STEP_COUNT = 16
    DEFAULT_TEMPO = 120
    DEFAULT_SWING = 0.0
    
    def __init__(self):
        """Initialize the sequencer with default values."""
        self.pattern: List[List[bool]] = [[False] * self.STEP_COUNT for _ in range(self.TRACK_COUNT)]
        self.tempo_bpm = self.DEFAULT_TEMPO
        self.swing_ratio = self.DEFAULT_SWING  # Stored as ratio (0.0 to 0.6 for 0-60%)
        self.current_step = 0
        self.is_running = False
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pattern_lock = threading.RLock()
    
    def start(self, tempo: int, swing: float, callback: Callable[[int], None], trigger: Callable[[int], None]) -> None:
        """
        Start the playback thread.
        
        Args:
            tempo: Tempo in BPM.
            swing: Swing in percentage (0-60).
            callback: Called each step for UI updates (advance step cursor).
            trigger: Called each step for audio triggering.
        """
        with self.pattern_lock:
            self.tempo_bpm = tempo
            self.swing_ratio = swing / 100.0  # Convert percentage to ratio
            self.current_step = 0
            self.is_running = True
            self.stop_event.clear()
        
        self.thread = threading.Thread(target=self._playback_thread, args=(callback, trigger), daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        """Stop the playback and join the thread safely."""
        self.is_running = False
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None
    
    def _playback_thread(self, callback: Callable[[int], None], trigger: Callable[[int], None]) -> None:
        """Background thread for playback timing loop."""
        step = 0
        while self.is_running and not self.stop_event.is_set():
            # Calculate duration for this step
            duration = self._calculate_step_duration(step)
            sleep(duration)
            
            if not self.is_running or self.stop_event.is_set():
                break
            
            with self.pattern_lock:
                self.current_step = step
            trigger(step)
            callback(step)
            
            step = (step + 1) % self.STEP_COUNT
    
    def _calculate_step_duration(self, step: int) -> float:
        """
        Calculate the duration for a given step considering tempo and swing.
        
        Args:
            step: The step index (0-15).
        
        Returns:
            Duration in seconds.
        """
        base_duration = 60.0 / (self.tempo_bpm * 4)  # 4 steps per beat
        swing = self.swing_ratio
        
        if step % 2 == 0:  # On-beat steps: longer duration
            return base_duration * (1 + swing)
        else:  # Off-beat steps: shorter duration
            return base_duration * (1 - swing)
    
    def toggle_step(self, track_index: int, step_index: int) -> bool:
        """
        Toggle a step's state. Helper for external access during live editing.
        
        Args:
            track_index: Index of the track (0-3).
            step_index: Index of the step (0-15).
        
        Returns:
            New state of the step.
        """
        with self.pattern_lock:
            self.pattern[track_index][step_index] = not self.pattern[track_index][step_index]
            return self.pattern[track_index][step_index]
    
    def get_pattern(self) -> List[List[bool]]:
        """Get a copy of the current pattern for serialization."""
        with self.pattern_lock:
            return [row[:] for row in self.pattern]
    
    def set_pattern(self, new_pattern: List[List[bool]]) -> None:
        """
        Set the pattern from a given grid.
        
        Args:
            new_pattern: 4x16 boolean grid.
        """
        with self.pattern_lock:
            for i in range(self.TRACK_COUNT):
                for j in range(self.STEP_COUNT):
                    self.pattern[i][j] = new_pattern[i][j]