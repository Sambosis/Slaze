from pathlib import Path
import random
import struct
import wave

def generate_snare_wav(output_path: Path, duration_seconds: float = 0.2, sample_rate: int = 44100, bit_depth: int = 16):
    """Generate a short noise burst WAV file for snare sample.

    Creates a mono 16-bit WAV file with white noise to simulate a snare drum.

    Args:
        output_path: Path to save the WAV file.
        duration_seconds: Length of the noise burst in seconds.
        sample_rate: Sample rate in Hz (default 44.1kHz).
        bit_depth: Bit depth (default 16-bit).
    """
    n_samples = int(sample_rate * duration_seconds)
    amp = 32767  # Max amplitude for 16-bit

    # Generate white noise
    noise = [random.randint(-amp, amp) for _ in range(n_samples)]

    # Apply a simple envelope or fade to simulate decay (optional)
    for i in range(n_samples):
        noise[i] *= max(0, 1.0 - i / n_samples)  # Linear fade out

    # Pack into bytes
    data = b''.join(struct.pack('<h', int(sample)) for sample in noise)

    # Write WAV file
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data)

# Generate the snare.wav file
asset_dir = Path(__file__).parent.parent / "assets" / "samples"
asset_dir.mkdir(parents=True, exist_ok=True)
snare_path = asset_dir / "snare.wav"
generate_snare_wav(snare_path)