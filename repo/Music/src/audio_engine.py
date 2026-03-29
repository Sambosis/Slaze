from __future__ import annotations

import logging
import shutil
import wave
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

try:
    import pygame
except ImportError as exc:
    pygame = None  # type: ignore[assignment]
    _PYGAME_IMPORT_ERROR = exc
else:
    _PYGAME_IMPORT_ERROR = None

if TYPE_CHECKING:
    from pygame.mixer import Channel, Sound
else:
    Channel = Any
    Sound = Any

LOGGER = logging.getLogger(__name__)

TRACK_NAMES: tuple[str, ...] = ("Kick", "Snare", "Hi-Hat", "Clap")
DEFAULT_SAMPLE_FILENAMES: tuple[str, ...] = ("kick.wav", "snare.wav", "hihat.wav", "clap.wav")

MIXER_FREQUENCY = 44_100
MIXER_SIZE = -16
MIXER_CHANNELS = 2
MIXER_BUFFER = 256

ASSETS_DIR = (Path(__file__).resolve().parent.parent / "assets" / "samples").resolve()

__all__ = ["AudioEngine", "load_default_samples", "TRACK_NAMES", "DEFAULT_SAMPLE_FILENAMES", "ASSETS_DIR"]


class AudioEngine:
    _initialized: bool = False

    def __init__(self, master_volume: float = 0.8) -> None:
        self._ensure_pygame_available()
        self._init_mixer()

        pygame.mixer.set_num_channels(len(TRACK_NAMES))
        self.channels: list[Channel] = [pygame.mixer.Channel(i) for i in range(len(TRACK_NAMES))]
        self.sample_map: Dict[str, Sound] = {}
        self.sample_paths: Dict[str, Path] = {}
        self.volume = 0.0
        self.set_master_volume(master_volume)

    def load_sample(self, track_name: str, file_path: Path) -> Path:
        canonical_name = _canonical_track_name(track_name)
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Sample '{path}' does not exist.")
        if path.suffix.lower() != ".wav":
            raise ValueError(f"Sample '{path.name}' must be a WAV file.")

        try:
            sound = pygame.mixer.Sound(path.as_posix())
        except pygame.error as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load WAV '{path.name}': {exc}") from exc

        self.sample_map[canonical_name] = sound
        self.sample_paths[canonical_name] = path
        LOGGER.info("Loaded sample '%s' for track '%s'.", path.name, canonical_name)
        return path

    def play_sample(self, track_name: str, channel_index: Optional[int] = None) -> None:
        if not self.sample_map:
            return

        canonical_name = _canonical_track_name(track_name)
        sound = self.sample_map.get(canonical_name)
        if sound is None:
            LOGGER.debug("No sample loaded for track '%s'.", canonical_name)
            return

        channel = self._resolve_channel(canonical_name, channel_index)
        if channel is None:
            LOGGER.warning("No available mixer channel for track '%s'.", canonical_name)
            return

        channel.set_volume(self.volume)
        channel.play(sound)

    def set_master_volume(self, volume: float) -> None:
        self._ensure_ready()
        clamped = max(0.0, min(1.0, float(volume)))
        self.volume = clamped
        for channel in self.channels:
            channel.set_volume(clamped)
        LOGGER.debug("Master volume set to %.2f", clamped)

    def quit_mixer(self) -> None:
        if pygame is None:
            return
        if pygame.mixer.get_init():
            pygame.mixer.stop()
            pygame.mixer.quit()
            AudioEngine._initialized = False
            LOGGER.info("Pygame mixer shut down.")

    def _ensure_ready(self) -> None:
        self._ensure_pygame_available()
        if not pygame.mixer.get_init():
            self._init_mixer()
            pygame.mixer.set_num_channels(len(TRACK_NAMES))
            self.channels = [pygame.mixer.Channel(i) for i in range(len(TRACK_NAMES))]

    def _resolve_channel(self, track_name: str, override_index: Optional[int]) -> Optional[Channel]:
        if override_index is not None and 0 <= override_index < len(self.channels):
            return self.channels[override_index]
        try:
            idx = TRACK_NAMES.index(track_name)
        except ValueError:
            return pygame.mixer.find_channel()
        return self.channels[idx]

    def _ensure_pygame_available(self) -> None:
        if pygame is None:
            raise RuntimeError(
                "pygame is required for audio playback but is not installed. "
                "Install it with 'pip install pygame'."
            ) from _PYGAME_IMPORT_ERROR

    def _init_mixer(self) -> None:
        if AudioEngine._initialized and pygame.mixer.get_init():
            return
        try:
            pygame.mixer.pre_init(
                frequency=MIXER_FREQUENCY,
                size=MIXER_SIZE,
                channels=MIXER_CHANNELS,
                buffer=MIXER_BUFFER,
            )
            pygame.mixer.init()
        except pygame.error as exc:  # pragma: no cover
            raise RuntimeError(f"Unable to initialize audio device: {exc}") from exc
        AudioEngine._initialized = True
        LOGGER.info(
            "Initialized mixer @ %d Hz, %d-bit, %d channels.",
            MIXER_FREQUENCY,
            abs(MIXER_SIZE),
            MIXER_CHANNELS,
        )


def load_default_samples(sample_dir: Path) -> Dict[str, Path]:
    target_dir = Path(sample_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, Path] = {}
    for track_name, filename in zip(TRACK_NAMES, DEFAULT_SAMPLE_FILENAMES):
        destination = target_dir / filename
        source = ASSETS_DIR / filename

        if not destination.exists():
            try:
                if source.exists():
                    if source.resolve() != destination.resolve():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, destination)
                else:
                    _write_silence_wav(destination)
                    LOGGER.warning(
                        "Bundled sample '%s' missing; generated placeholder at '%s'.",
                        filename,
                        destination,
                    )
            except OSError as exc:
                LOGGER.error("Failed to prepare sample '%s': %s", filename, exc)
                raise
        mapping[track_name] = destination

    return mapping


def _write_silence_wav(path: Path, duration_seconds: float = 0.2) -> None:
    frame_rate = MIXER_FREQUENCY
    n_frames = max(int(frame_rate * duration_seconds), frame_rate // 20)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path.as_posix(), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)


def _canonical_track_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    aliases = {
        "bd": "kick",
        "bassdrum": "kick",
        "bass-drum": "kick",
        "hh": "hi-hat",
        "hihat": "hi-hat",
        "hi hat": "hi-hat",
    }
    normalized = aliases.get(normalized, normalized)
    for track in TRACK_NAMES:
        if normalized == track.lower():
            return track
    raise ValueError(f"Unknown track name '{name}'. Expected one of {TRACK_NAMES}.")