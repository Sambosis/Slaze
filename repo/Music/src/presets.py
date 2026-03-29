from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Mapping, TypedDict, Union

__all__ = ["PresetData", "PresetManager", "_validate_preset_data"]

TRACK_COUNT = 4
STEP_COUNT = 16
TEMPO_RANGE = (60, 200)
SWING_RANGE = (0.0, 60.0)
VOLUME_RANGE = (0.0, 100.0)
JSON_EXTENSION = ".json"

PathLike = Union[str, Path]


class PresetData(TypedDict):
    tempo: int
    swing: float
    volume: float
    pattern: List[List[int]]
    samples: List[str]


class PresetManager:
    def __init__(self, directory: PathLike) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, filename: PathLike, data: Mapping[str, Any]) -> Path:
        path = self._resolve_filename(filename)
        preset = _validate_preset_data(data)
        payload = {
            "tempo": preset["tempo"],
            "swing": preset["swing"],
            "volume": preset["volume"],
            "pattern": preset["pattern"],
            "samples": preset["samples"],
        }

        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            tmp_path.replace(path)
        except OSError as exc:
            raise OSError(f"Failed to save preset '{path.name}': {exc}") from exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        return path

    def load(self, filename: PathLike) -> PresetData:
        path = self._resolve_filename(filename)
        if not path.exists():
            raise FileNotFoundError(f"Preset file '{path.name}' does not exist in '{self.directory}'.")
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Preset file '{path.name}' contains invalid JSON: {exc}") from exc
        except OSError as exc:
            raise OSError(f"Failed to read preset '{path.name}': {exc}") from exc

        return _validate_preset_data(raw)

    def list_presets(self) -> List[Path]:
        return sorted(self.directory.glob(f"*{JSON_EXTENSION}"))

    def _resolve_filename(self, filename: PathLike) -> Path:
        name_str = str(filename).strip()
        if not name_str:
            raise ValueError("Preset filename cannot be empty.")

        candidate = Path(filename).expanduser()
        if candidate.suffix.lower() != JSON_EXTENSION:
            candidate = candidate.with_suffix(JSON_EXTENSION)

        if not candidate.is_absolute():
            candidate = (self.directory / candidate).resolve()
        else:
            candidate = candidate.resolve()

        base_dir = self.directory.resolve()
        try:
            candidate.relative_to(base_dir)
        except ValueError as exc:
            raise ValueError(f"Preset path '{candidate}' must reside within '{base_dir}'.") from exc

        if candidate.is_dir():
            raise ValueError("Preset filename points to a directory, not a file.")

        return candidate


def _validate_preset_data(data: Any) -> PresetData:
    if not isinstance(data, Mapping):
        raise ValueError("Preset data must be a mapping.")

    tempo = _coerce_int_field(data, "tempo", TEMPO_RANGE)
    swing = _coerce_float_field(data, "swing", SWING_RANGE)
    volume = _coerce_float_field(data, "volume", VOLUME_RANGE)
    pattern = _coerce_pattern_field(data, "pattern")
    samples = _coerce_samples_field(data, "samples")

    return PresetData(
        tempo=tempo,
        swing=swing,
        volume=volume,
        pattern=pattern,
        samples=samples,
    )


def _coerce_int_field(data: Mapping[str, Any], name: str, bounds: tuple[int, int]) -> int:
    if name not in data:
        raise ValueError(f"Preset is missing required field '{name}'.")
    number = _to_float(data[name], name)
    integer = int(round(number))
    if abs(integer - number) > 1e-6:
        raise ValueError(f"Preset field '{name}' must be an integer.")
    lower, upper = bounds
    if not (lower <= integer <= upper):
        raise ValueError(f"Preset field '{name}' must be between {lower} and {upper}.")
    return integer


def _coerce_float_field(data: Mapping[str, Any], name: str, bounds: tuple[float, float]) -> float:
    if name not in data:
        raise ValueError(f"Preset is missing required field '{name}'.")
    number = _to_float(data[name], name)
    lower, upper = bounds
    if not (lower <= number <= upper):
        raise ValueError(f"Preset field '{name}' must be between {lower} and {upper}.")
    return float(number)


def _coerce_pattern_field(data: Mapping[str, Any], name: str) -> List[List[int]]:
    if name not in data:
        raise ValueError(f"Preset is missing required field '{name}'.")
    raw_pattern = data[name]
    if not isinstance(raw_pattern, list):
        raise ValueError("Preset field 'pattern' must be a list of track rows.")
    if len(raw_pattern) != TRACK_COUNT:
        raise ValueError(f"Preset pattern must contain exactly {TRACK_COUNT} tracks.")
    pattern: List[List[int]] = []
    for track_index, row in enumerate(raw_pattern):
        if not isinstance(row, list):
            raise ValueError(f"Pattern track {track_index + 1} must be a list of steps.")
        if len(row) != STEP_COUNT:
            raise ValueError(f"Pattern track {track_index + 1} must contain {STEP_COUNT} steps.")
        steps: List[int] = []
        for step_index, step in enumerate(row):
            steps.append(_coerce_step_value(step, track_index, step_index))
        pattern.append(steps)
    return pattern


def _coerce_samples_field(data: Mapping[str, Any], name: str) -> List[str]:
    if name not in data:
        raise ValueError(f"Preset is missing required field '{name}'.")
    raw_samples = data[name]
    if not isinstance(raw_samples, list):
        raise ValueError("Preset field 'samples' must be a list of strings.")
    if len(raw_samples) != TRACK_COUNT:
        raise ValueError(f"Preset samples list must contain {TRACK_COUNT} entries.")
    samples: List[str] = []
    for index, entry in enumerate(raw_samples):
        if entry is None:
            samples.append("")
        else:
            samples.append(str(entry))
    return samples


def _coerce_step_value(value: Any, track_index: int, step_index: int) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        number = float(value)
        if abs(number) <= 1e-6:
            return 0
        if abs(number - 1.0) <= 1e-6:
            return 1
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"0", "1"}:
            return int(stripped)
    raise ValueError(
        f"Pattern value at track {track_index + 1}, step {step_index + 1} must be 0 or 1."
    )


def _to_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Preset field '{field_name}' must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Preset field '{field_name}' must be numeric.") from exc