"""
recorder.py

Reusable video recorder for Pygame projects using imageio-ffmpeg.

Features:
- Simple API: Recorder(path).capture(surface_or_none)
- Works with pygame.Surface or numpy arrays
- Auto-detect display surface when capture() called with no args
- Sensible default FPS (config.FIXED_TIMESTEP if available, else 60)
- Auto-create output directories and timestamped filenames
- Context manager support (with Recorder(...) as r: ...)
- Pause/resume/toggle recording and frame skipping
- Configurable codec, quality, bitrate, and extra ffmpeg params

Example usage:

    import pygame
    from recorder import Recorder

    pygame.init()
    screen = pygame.display.set_mode((800, 600))

    with Recorder("videos/demo.mp4") as rec:  # auto-mkdir, sensible default fps
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((30, 30, 30))
            # ... draw your game ...

            pygame.display.flip()
            rec.capture(screen)  # or rec.capture() to grab current display
            clock.tick(60)

"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Optional, Sequence, Union

import numpy as np
import imageio

try:  # Optional pygame import for type hints and conversions
    import pygame
except Exception:  # pragma: no cover - allow using arrays without pygame installed
    pygame = None  # type: ignore


def _get_default_fps() -> int:
    """Choose a sensible default FPS.

    If a project-level `config` module with `FIXED_TIMESTEP` exists, use its
    inverse as the default fps. Otherwise, fallback to 60.
    """
    try:
        import config  # type: ignore

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
    """Video recorder for Pygame surfaces and numpy frames.

    Parameters
    - path: File path or directory. If a directory is given, an auto-named file
      will be created inside it. If omitted/None, defaults to videos/<timestamp>.mp4
    - fps: Target frames per second. If None, inferred from config or defaults to 60.
    - codec: FFmpeg codec (e.g., 'libx264').
    - quality: Image quality for some codecs (1=best, 10=worst). Ignored if bitrate set.
    - bitrate: Explicit bitrate string like '4M'. Overrides quality when provided.
    - macro_block_size: Set 1 to allow any size. Higher values enforce multiples.
    - ffmpeg_params: Extra ffmpeg args as a list, e.g., ['-pix_fmt', 'yuv420p'].
    - auto_mkdir: Create parent directories as needed.
    - frame_skip: Write every (frame_skip + 1)-th frame.
    - start_paused: Start in paused mode (no frames written until resume()).
    - resize_to: Optional (width, height) to resize frames before writing.

    Methods
    - capture(surface_or_array=None): Append a frame. If None, grabs display surface.
    - capture_frame(ndarray): Append HxWx3 uint8 RGB array directly.
    - record_display(): Convenience to capture current display surface.
    - pause()/resume()/toggle(): Control recording without recreating the writer.
    - hook_display_flip(): Context manager to auto-capture on pygame.display.flip().
    - close(): Finish and close the file. Also via context manager.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        fps: Optional[int] = None,
        *,
        codec: str = "libx264",
        quality: int = 5,
        bitrate: Optional[str] = None,
        macro_block_size: int = 1,
        ffmpeg_params: Optional[Sequence[str]] = None,
        auto_mkdir: bool = True,
        frame_skip: int = 0,
        start_paused: bool = False,
        resize_to: Optional[tuple[int, int]] = None,
    ) -> None:
        # Resolve output path and ensure directory exists
        if path is None or os.path.isdir(path):
            base_dir = path if path else "videos"
            path = default_output_path(base_dir=base_dir)
        parent = os.path.dirname(path)
        if auto_mkdir and parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        # Determine FPS
        self.fps: int = int(fps) if fps else _get_default_fps()

        # Open video writer
        writer_kwargs = {
            "fps": self.fps,
            "codec": codec,
            "macro_block_size": macro_block_size,
        }
        if bitrate:
            writer_kwargs["bitrate"] = bitrate
        else:
            writer_kwargs["quality"] = quality
        # Pass through custom ffmpeg params if provided. We no longer force a
        # default "-pix_fmt yuv420p" here because imageio-ffmpeg already sets
        # a sensible pixel format for libx264; adding our own caused a noisy
        # warning about multiple -pix_fmt options. Users needing a specific
        # pixel format can supply it explicitly via ffmpeg_params.
        if ffmpeg_params is not None:
            writer_kwargs["ffmpeg_params"] = list(ffmpeg_params)

        self._writer = imageio.get_writer(path, **writer_kwargs)
        self._path = path
        self._frame_index = 0
        self._frame_skip = max(0, int(frame_skip))
        self._paused = bool(start_paused)
        self._resize_to = resize_to

    # -------------------------
    # Context manager interface
    # -------------------------
    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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
        """Capture and append a frame.

        - If `surface_or_array` is a pygame.Surface, it will be converted to HxWx3 RGB.
        - If it's an ndarray, it must be HxWx3 or WxHx3 and uint8. WxHx3 will be transposed.
        - If None, attempts to capture from `pygame.display.get_surface()`.

        Returns the global frame index after processing this call (whether written or skipped).
        """
        self._frame_index += 1
        if self._paused:
            return self._frame_index

        # Apply frame skipping
        if self._frame_skip and (self._frame_index - 1) % (self._frame_skip + 1) != 0:
            return self._frame_index

        frame: Optional[np.ndarray] = None

        if surface_or_array is None:
            if pygame is None:
                raise RuntimeError("pygame is not available; provide a numpy frame to capture().")
            display_surface = pygame.display.get_surface()
            if display_surface is None:
                raise RuntimeError("No active display surface; provide a surface or numpy frame.")
            frame = self._surface_to_frame(display_surface)
        elif pygame is not None and isinstance(surface_or_array, pygame.Surface):
            frame = self._surface_to_frame(surface_or_array)
        else:
            frame = self._array_to_frame(surface_or_array)  # type: ignore[arg-type]

        self._writer.append_data(frame)
        return self._frame_index

    def record_display(self) -> int:
        """Convenience to capture the current display surface."""
        return self.capture(None)

    def capture_frame(self, frame: np.ndarray) -> int:
        """Append a validated HxWx3 uint8 RGB array directly."""
        return self.capture(frame)

    @contextmanager
    def hook_display_flip(self):
        """Auto-capture on every pygame.display.flip() within the context.

        Example:
            with Recorder("videos/run.mp4").hook_display_flip() as rec:
                while running:
                    ...
                    pygame.display.flip()  # automatically captured
        """
        if pygame is None:
            raise RuntimeError("pygame is not available to hook display flip.")
        original_flip = pygame.display.flip

        def wrapped_flip(*args, **kwargs):
            result = original_flip(*args, **kwargs)
            try:
                self.capture(None)
            except Exception:
                pass
            return result

        pygame.display.flip = wrapped_flip  # type: ignore[assignment]
        try:
            yield self
        finally:
            pygame.display.flip = original_flip  # type: ignore[assignment]

    # -------------------------
    # Conversion helpers
    # -------------------------
    def _surface_to_frame(self, surface: "pygame.Surface") -> np.ndarray:
        # pygame.surfarray.array3d returns (W, H, 3) in RGB, we need (H, W, 3)
        if self._resize_to is not None and pygame is not None:
            surface = pygame.transform.smoothscale(surface, self._resize_to)
        arr = pygame.surfarray.array3d(surface)  # type: ignore[attr-defined]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected a 3-channel RGB surface")
        frame = np.transpose(arr, (1, 0, 2))
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        return frame

    def _array_to_frame(self, array_like: np.ndarray) -> np.ndarray:
        if not isinstance(array_like, np.ndarray):
            raise TypeError("capture() expects a pygame.Surface or numpy.ndarray")
        arr = array_like
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected an array of shape HxWx3 or WxHx3 with 3 channels")
        # If WxHx3, transpose to HxWx3
        if arr.shape[0] < 10 and arr.shape[1] > 10:
            # Heuristic isn't great; just check if shape matches a common surface pattern
            pass
        # Normalize to HxWx3
        if arr.shape[0] in (3,) and arr.shape[-1] not in (3,):
            raise ValueError("Ambiguous array shape; expected channels-last RGB")
        # If looks like WxHx3 (as pygame array3d), transpose to HxWx3
        if arr.shape[0] != arr.shape[1] and arr.shape[0] < arr.shape[1]:
            # Use heuristic if width != height and likely (W,H,3)
            possible = np.transpose(arr, (1, 0, 2))
            if possible.ndim == 3 and possible.shape[2] == 3:
                arr = possible
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if self._resize_to is not None and pygame is not None:
            # Resize via pygame for quality
            surf = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))  # back to (W,H,3)
            surf = pygame.transform.smoothscale(surf, self._resize_to)
            arr = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
        return arr

    # -------------------------
    # Cleanup
    # -------------------------
    def close(self) -> None:
        if getattr(self, "_writer", None) is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None  # type: ignore[assignment]


@contextmanager
def record_pygame(
    path: Optional[str] = None,
    fps: Optional[int] = None,
    **kwargs,
):
    """Convenience: create a Recorder and hook pygame.display.flip().

    Usage:
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