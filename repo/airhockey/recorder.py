"""
recorder.py

Reusable, high-performance video recorder for Pygame projects using imageio-ffmpeg.

Features:
- Asynchronous encoding: writing frames doesn't block the main game loop.
- Graceful error recovery: encoding failures won't crash your game process.
- Safe termination: atexit handlers ensure videos are finalized even on crash.
- Thread-safe: uses threading.Event for cross-thread state, compatible with
  free-threaded Python (3.13+ --disable-gil).
- Simple API: Recorder(path).capture(surface_or_none)
- Works with pygame.Surface or numpy arrays
- Auto-detect display surface when capture() called with no args
- Sensible default FPS (config.FIXED_TIMESTEP if available, else 60)
- Auto-create output directories and timestamped filenames
- Context manager support (with Recorder(...) as r: ...)
- Pause/resume/toggle recording and frame skipping
- Configurable codec, quality, bitrate, and extra ffmpeg params
- Lossless recording preset (``lossless=True``)
- Crisp pixel-art scaling (``smooth_scale=False``)
- Auto-detect codec from file extension (.mp4, .gif, .avi, .webm)
- Live frame rate tracking via actual_fps property

Example usage:

    import pygame
    from recorder import Recorder

    pygame.init()
    screen = pygame.display.set_mode((800, 600))

    with Recorder("videos/demo.mp4") as rec:
        running = True
        clock = pygame.time.Clock()
        while running:
            # ... draw your game ...
            pygame.display.flip()
            rec.capture(screen)  # background thread handles the encoding
            clock.tick(60)
"""
from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
import time
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import imageio

if TYPE_CHECKING:
    import pygame as _pygame_type
    PygameSurface = _pygame_type.Surface
else:
    PygameSurface = None

# Runtime pygame import – optional dependency
try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)

# Sentinel to detect when the user did NOT explicitly pass a codec
_CODEC_AUTO = object()

# Extension → codec mapping for auto-detection
_EXT_CODEC_MAP: dict[str, str] = {
    ".mp4": "libx264",
    ".gif": "gif",
    ".avi": "rawvideo",
    ".webm": "libvpx",
    ".mkv": "libx264",
    ".mov": "libx264",
}


def _get_default_fps() -> int:
    """Choose a sensible default FPS."""
    try:
        import config  # type: ignore[import-untyped]

        if hasattr(config, "FIXED_TIMESTEP") and config.FIXED_TIMESTEP > 0:
            return int(round(1.0 / float(config.FIXED_TIMESTEP)))
        if hasattr(config, "FPS") and int(config.FPS) > 0:
            return int(config.FPS)
    except Exception:
        pass
    return 60


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def default_output_path(
    base_dir: str = "videos",
    prefix: str = "game",
    ext: str = "mp4",
) -> str:
    """Build a default output path like videos/game_YYYYmmdd_HHMMSS.mp4."""
    filename = f"{prefix}_{_timestamp()}.{ext}"
    return os.path.join(base_dir, filename)


ArrayLike = Union[np.ndarray, "pygame.Surface"]


class Recorder:
    """
    High-performance video recorder for Pygame surfaces and numpy frames.
    Uses a background thread to prevent encoding from stalling the main game loop.

    Parameters
    - path: File path or directory.
    - fps: Target frames per second. If None, inferred from config or defaults to 60.
    - codec: FFmpeg codec (e.g., 'libx264'). Auto-detected from extension if omitted.
    - quality: Image quality for some codecs (1=best, 10=worst). Ignored if bitrate set.
    - bitrate: Explicit bitrate string like '4M'. Overrides quality when provided.
    - lossless: If True, overrides codec/params for mathematically lossless video
      (libx264rgb with -crf 0). Any additional ``ffmpeg_params`` are merged.
    - macro_block_size: Set 1 to allow any size. Higher values enforce multiples.
    - ffmpeg_params: Extra ffmpeg args as a list, e.g., ['-pix_fmt', 'yuv420p'].
      These are merged (not replaced) even when ``lossless=True``.
    - auto_mkdir: Create parent directories as needed.
    - frame_skip: Write every (frame_skip + 1)-th frame. For example, ``frame_skip=0``
      writes every frame (default), ``frame_skip=1`` writes every 2nd frame.
    - start_paused: Start in paused mode (no frames written until resume()).
    - resize_to: Optional (width, height) to resize frames before writing.
    - smooth_scale: If True (default), resizes using bilinear interpolation. If False,
      uses nearest-neighbor (ideal for crisp pixel-art).

    Properties
    - path: Output file path.
    - frame_index: Total frames processed (including skipped/paused).
    - frames_written: Total frames actually written to the video file.
    - duration: Estimated video duration in seconds (frames_written / fps).
    - actual_fps: Measured frames-per-second based on wall-clock time.
    - paused: Whether recording is currently paused.
    - is_open: Whether the writer is still open.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        fps: Optional[int] = None,
        *,
        codec: object = _CODEC_AUTO,
        quality: int = 5,
        bitrate: Optional[str] = None,
        lossless: bool = False,
        macro_block_size: int = 1,
        ffmpeg_params: Optional[Sequence[str]] = None,
        auto_mkdir: bool = True,
        frame_skip: int = 0,
        start_paused: bool = False,
        resize_to: Optional[tuple[int, int]] = None,
        smooth_scale: bool = True,
    ) -> None:
        # Resolve output path
        if path is None or os.path.isdir(path):
            base_dir = path if path else "videos"
            path = default_output_path(base_dir=base_dir)
        parent = os.path.dirname(path)
        if auto_mkdir and parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        self.fps: int = int(fps) if fps else _get_default_fps()
        self._path = path

        # Determine FFmpeg kwargs
        writer_kwargs: dict = {
            "fps": self.fps,
            "macro_block_size": macro_block_size,
        }

        # Lossless override vs normal config
        if lossless:
            resolved_codec = "libx264rgb"
            # Start with lossless params, then merge any user-provided extras
            combined_params = ["-crf", "0"]
            if ffmpeg_params is not None:
                combined_params.extend(ffmpeg_params)
            writer_kwargs["ffmpeg_params"] = combined_params
        else:
            if codec is _CODEC_AUTO:
                ext = os.path.splitext(path)[1].lower()
                resolved_codec = _EXT_CODEC_MAP.get(ext, "libx264")
            else:
                resolved_codec = str(codec)

            if bitrate:
                writer_kwargs["bitrate"] = bitrate
            else:
                writer_kwargs["quality"] = quality

            if ffmpeg_params is not None:
                writer_kwargs["ffmpeg_params"] = list(ffmpeg_params)

        writer_kwargs["codec"] = resolved_codec
        self._codec = resolved_codec

        # Threading and IO components
        self._writer = imageio.get_writer(path, **writer_kwargs)
        self._queue: queue.Queue = queue.Queue(maxsize=120)  # Bound to prevent memory leak
        self._broken_event = threading.Event()  # Thread-safe broken flag
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="Recorder-Writer",
        )

        # State tracking
        self._frame_index = 0
        self._frames_written = 0
        self._frame_skip = max(0, int(frame_skip))
        self._paused = bool(start_paused)
        self._resize_to = resize_to
        self._smooth_scale = bool(smooth_scale)
        self._start_time: Optional[float] = None

        # Start background worker and register safe cleanup
        self._thread.start()
        atexit.register(self.close)

    def _worker(self) -> None:
        """Background thread: consume frames from the queue and write to disk.

        On error, drains remaining queued frames (calling task_done for each)
        so that close() never deadlocks waiting on queue.join().
        """
        try:
            while True:
                frame = self._queue.get()
                if frame is None:  # Sentinel value signaling shutdown
                    self._queue.task_done()
                    break
                self._writer.append_data(frame)
                self._queue.task_done()
        except Exception as e:
            self._broken_event.set()
            _log.error("Recorder background thread failed: %s", e)
            warnings.warn(f"Video encoding failed asynchronously: {e}", RuntimeWarning)
            # Drain any remaining items so close() doesn't deadlock
            try:
                while True:
                    self._queue.get_nowait()
                    self._queue.task_done()
            except queue.Empty:
                pass

    # -------------------------
    # Context manager interface
    # -------------------------
    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -------------------------
    # Repr
    # -------------------------
    def __repr__(self) -> str:
        broken = self._broken_event.is_set()
        return (
            f"Recorder(path={self._path!r}, fps={self.fps}, "
            f"paused={self._paused}, frames={self._frames_written}, "
            f"open={self.is_open}, broken={broken})"
        )

    # -------------------------
    # Destructor safety net
    # -------------------------
    def __del__(self) -> None:
        try:
            if hasattr(self, "_writer") and self._writer is not None:
                self.close()
        except Exception:
            pass  # Never raise from __del__

    # -------------------------
    # Properties
    # -------------------------
    @property
    def path(self) -> str:
        return self._path

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def frames_written(self) -> int:
        """Total number of frames actually written to the video file."""
        return self._frames_written

    @property
    def duration(self) -> float:
        """Estimated video duration in seconds (frames_written / fps)."""
        return self._frames_written / self.fps if self.fps > 0 else 0.0

    @property
    def actual_fps(self) -> float:
        """Measured capture rate based on wall-clock time since first frame.

        Returns 0.0 if no frames have been written yet.
        """
        if self._frames_written == 0 or self._start_time is None:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._frames_written / elapsed

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def is_open(self) -> bool:
        return self._writer is not None

    # -------------------------
    # Control
    # -------------------------
    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def toggle(self) -> None:
        self._paused = not self._paused

    # -------------------------
    # Capture methods
    # -------------------------
    def capture(self, surface_or_array: Optional[ArrayLike] = None) -> int:
        """Capture and asynchronously append a frame.

        - If ``surface_or_array`` is a ``pygame.Surface``, it will be converted to
          HxWx3 RGB via ``pygame.surfarray.array3d`` (which makes a full copy of the
          pixel data, ensuring the frame is safely enqueued even if the surface is
          modified on the next tick).
        - If it's an ndarray, it must be HxWx3 uint8.
        - If None, attempts to capture from ``pygame.display.get_surface()``.

        Returns the global frame index. If the recorder is broken or closed,
        fails gracefully and returns the index silently.
        """
        if not self.is_open or self._broken_event.is_set():
            return self._frame_index

        self._frame_index += 1
        if self._paused:
            return self._frame_index

        # Apply frame skipping
        if self._frame_skip and (self._frame_index - 1) % (self._frame_skip + 1) != 0:
            return self._frame_index

        frame: Optional[np.ndarray] = None

        try:
            if surface_or_array is None:
                if pygame is None:
                    raise RuntimeError("pygame is not available; provide a numpy frame.")
                display_surface = pygame.display.get_surface()
                if display_surface is None:
                    raise RuntimeError("No active display surface.")
                frame = self._surface_to_frame(display_surface)
            elif pygame is not None and isinstance(surface_or_array, pygame.Surface):
                frame = self._surface_to_frame(surface_or_array)
            else:
                frame = self._array_to_frame(surface_or_array)  # type: ignore[arg-type]

            # Track wall-clock start for actual_fps
            if self._start_time is None:
                self._start_time = time.monotonic()

            # Pass to background thread (timeout prevents infinite hang)
            self._queue.put(frame, timeout=2.0)
            self._frames_written += 1

        except queue.Full:
            self._broken_event.set()
            _log.error("Recorder queue full: encoding cannot keep up with game loop. Disabling.")
            warnings.warn("Video encoding too slow; recorder disabled.", RuntimeWarning)
        except Exception as e:
            self._broken_event.set()
            _log.error("Recorder capture failed: %s", e)
            warnings.warn(f"Capture failed: {e}. Recorder disabled.", RuntimeWarning)

        return self._frame_index

    def record_display(self) -> int:
        """Convenience to capture the current display surface."""
        return self.capture(None)

    def capture_frame(self, frame: np.ndarray) -> int:
        """Append a validated HxWx3 uint8 RGB array directly."""
        return self.capture(frame)

    @contextmanager
    def hook_display_flip(self):
        """Auto-capture on every ``pygame.display.flip()`` and ``pygame.display.update()``
        within the context.

        The recorder is also closed when the context exits, so this pattern is
        safe even without wrapping in a separate ``with`` block::

            with Recorder("videos/run.mp4").hook_display_flip() as rec:
                while running:
                    ...
                    pygame.display.flip()  # automatically captured
            # rec is closed here
        """
        if pygame is None:
            raise RuntimeError("pygame not available to hook display flip.")
        original_flip = pygame.display.flip
        original_update = pygame.display.update

        def wrapped_flip(*args, **kwargs):
            result = original_flip(*args, **kwargs)
            try:
                self.capture(None)
            except Exception as exc:
                warnings.warn(
                    f"Recorder failed during display flip: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return result

        def wrapped_update(*args, **kwargs):
            result = original_update(*args, **kwargs)
            try:
                self.capture(None)
            except Exception as exc:
                warnings.warn(
                    f"Recorder failed during display update: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return result

        pygame.display.flip = wrapped_flip  # type: ignore[assignment]
        pygame.display.update = wrapped_update  # type: ignore[assignment]
        try:
            yield self
        finally:
            pygame.display.flip = original_flip  # type: ignore[assignment]
            pygame.display.update = original_update  # type: ignore[assignment]
            self.close()

    # -------------------------
    # Conversion helpers
    # -------------------------
    def _apply_resize(self, surface: "pygame.Surface") -> "pygame.Surface":
        """Resize a surface using the configured scaling algorithm."""
        if self._resize_to is not None and pygame is not None:
            if self._smooth_scale:
                return pygame.transform.smoothscale(surface, self._resize_to)
            else:
                return pygame.transform.scale(surface, self._resize_to)
        return surface

    def _surface_to_frame(self, surface: "pygame.Surface") -> np.ndarray:
        """Convert a pygame.Surface to an HxWx3 uint8 numpy array."""
        surface = self._apply_resize(surface)
        # array3d returns (W, H, 3); the copy is intentional—it snapshots the
        # surface so the background thread can safely encode it while the game
        # loop mutates the original surface on the next tick.
        arr = pygame.surfarray.array3d(surface)  # type: ignore[attr-defined]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected a 3-channel RGB surface")
        frame = np.transpose(arr, (1, 0, 2))
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        return frame

    def _array_to_frame(self, array_like: np.ndarray) -> np.ndarray:
        """Validate and normalize a caller-provided numpy array to HxWx3 uint8."""
        if not isinstance(array_like, np.ndarray):
            raise TypeError("capture() expects a pygame.Surface or numpy.ndarray")
        arr = array_like
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"Expected an array of shape (H, W, 3), got {arr.shape}. "
                "The array must have 3 dimensions with 3 channels in the last axis."
            )

        if arr.shape[0] != arr.shape[1] and arr.shape[0] < arr.shape[1]:
            warnings.warn(
                f"Array shape {arr.shape} looks like (W, H, 3) — transposing to "
                f"(H, W, 3). For best results, pass arrays in (H, W, 3) directly.",
                stacklevel=3,
            )
            arr = np.transpose(arr, (1, 0, 2))

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        if self._resize_to is not None and pygame is not None:
            surf = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
            surf = self._apply_resize(surf)
            arr = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
        return arr

    # -------------------------
    # Cleanup
    # -------------------------
    def close(self) -> None:
        """Wait for pending frames to write, close the file, and deregister atexit."""
        if getattr(self, "_writer", None) is not None:

            # Unregister atexit safety to avoid double-closure
            try:
                atexit.unregister(self.close)
            except AttributeError:
                pass  # safety check

            # Flush queue and shutdown thread safely
            thread = getattr(self, "_thread", None)
            if thread is not None and thread.is_alive():
                if not self._broken_event.is_set():
                    # Normal path: signal the worker to stop
                    try:
                        self._queue.put(None, timeout=2.0)
                    except queue.Full:
                        _log.warning("Recorder queue full during close; draining.")
                        # Drain one item to make room for sentinel
                        try:
                            self._queue.get_nowait()
                            self._queue.task_done()
                            self._queue.put(None, timeout=1.0)
                        except (queue.Empty, queue.Full):
                            pass
                # Wait for thread regardless of broken state
                thread.join(timeout=5.0)
                if thread.is_alive():
                    _log.warning("Recorder worker thread did not exit in time.")

            # Finalize file
            try:
                self._writer.close()
            except Exception as e:
                _log.error("Recorder writer failed to close: %s", e)
            finally:
                self._writer = None  # type: ignore[assignment]

            # Log summary
            vid_duration = self._frames_written / self.fps if self.fps > 0 else 0.0
            _log.info(
                "Recorder closed: %d frames, ~%.1fs @ %d fps → %s",
                self._frames_written,
                vid_duration,
                self.fps,
                self._path,
            )


@contextmanager
def record_pygame(
    path: Optional[str] = None,
    fps: Optional[int] = None,
    **kwargs,
):
    """Convenience: create a Recorder and hook ``pygame.display.flip()``.

    The recorder is automatically closed when the context exits::

        with record_pygame("videos/run.mp4", fps=60) as rec:
            while running:
                ...
                pygame.display.flip()
        # rec is closed automatically
    """
    rec = Recorder(path=path, fps=fps, **kwargs)
    with rec.hook_display_flip() as hooked:
        yield hooked


__all__ = ["Recorder", "default_output_path", "record_pygame"]
