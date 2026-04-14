# Recorder (`recorder.py`)

A high-performance, production-grade video recording utility for Pygame projects. Uses a **background thread** to encode frames asynchronously, so your game loop never stutters during recording.

## Key Features

| Feature | Description |
|---|---|
| **Async Encoding** | Background thread handles FFmpeg I/O — `capture()` is non-blocking |
| **Process Safety** | `atexit` handler finalizes videos even on `Ctrl+C` or crashes |
| **Error Recovery** | Disk full / pipe break → graceful disable, no game crash |
| **Thread-Safe** | Uses `threading.Event`; compatible with free-threaded Python 3.13+ |
| **Codec Auto-Detect** | `.mp4`→libx264, `.gif`→gif, `.avi`→rawvideo, `.webm`→libvpx |
| **Lossless Preset** | `lossless=True` for pixel-perfect RL debugging |
| **Pixel-Art Scaling** | `smooth_scale=False` for crisp nearest-neighbor upscaling |
| **Smart Defaults** | Auto-detects FPS from `config.FIXED_TIMESTEP` or `config.FPS` |

## Properties

| Property | Type | Description |
|---|---|---|
| `path` | `str` | Output file path |
| `fps` | `int` | Target frames per second |
| `frame_index` | `int` | Total frames processed (including skipped/paused) |
| `frames_written` | `int` | Frames actually written to the video file |
| `duration` | `float` | Estimated video duration in seconds (`frames_written / fps`) |
| `actual_fps` | `float` | Measured capture rate based on wall-clock time |
| `paused` | `bool` | Whether recording is currently paused |
| `is_open` | `bool` | Whether the writer is still open |

## Usage

### Basic Manual Capture

```python
import pygame
from recorder import Recorder

pygame.init()
screen = pygame.display.set_mode((800, 600))

with Recorder(fps=60) as rec:
    running = True
    clock = pygame.time.Clock()
    while running:
        screen.fill((30, 30, 30))
        # ... draw your game ...
        pygame.display.flip()
        rec.capture(screen)  # ~0ms, queued to background thread
        clock.tick(60)
    print(f"{rec.frames_written} frames, {rec.duration:.1f}s video")
```

### Auto-Hook Game Loop

```python
from recorder import record_pygame

# Intercepts pygame.display.flip() — zero code changes in your loop
with record_pygame("videos/gameplay.mp4", fps=60):
    while running:
        # ... your game loop ...
        pygame.display.flip()  # automatically captured + closed on exit
```

### Lossless & Pixel-Art

```python
from recorder import Recorder

# Pixel-perfect, no compression artifacts, crisp upscaling
with Recorder("videos/debug.mp4", lossless=True, smooth_scale=False, resize_to=(800, 600)) as rec:
    # ... capture RL environment frames ...
    pass
```

### GIF Recording

```python
# Codec auto-detected from extension
with Recorder("videos/preview.gif", fps=15) as rec:
    # ... capture frames ...
    pass
```

### Logging Summary

`close()` logs a summary via Python's `logging` module (logger name: `recorder`):

```python
import logging
logging.basicConfig(level=logging.INFO)
# On close: "Recorder closed: 300 frames, ~5.0s @ 60 fps → videos/demo.mp4"
```
