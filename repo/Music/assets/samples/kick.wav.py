"""
Binary placeholder for the default kick drum sample.

This module contains a function to generate the WAV data recursively for a short 44.13
kHz 16-bit mono WAV producing a simple low-frequency thump, simulating a kick drum sound.
"""

import struct
import math


def generate_kick_wav() -> bytes:
    """
    Generate the binary data for a simple kick drum WAV.

    This creates a 0.2-second wave at 44.1 kHz, 16-bit mono, with a low-frequency
    sine wave (approximately 60 Hz) that decays exponentially to simulate a thump.

    Returns:
        The complete WAV file data as bytes, ready to be written to a file.
    """
    sample_rate = 44100
    duration = 0.2  # Seconds
    num_samples = int(sample_rate * duration)
    freq = 60  # Hertz for low-frequency thump
    decay_rate = 10.0  # Exponential decay rate

    frames = []
    for i in range(num_samples):
        t = i / sample_rate
        wave = math.sin(2 * math.pi * freq * t) * math.exp(-t * decay_rate)
        sample = int(wave * 28000)  # Amplitude scaled to avoid clipping in 16-bit signed
        frames.append(sample)

    # Pack frames into binary data
    data = b''.join(s.to_bytes(2, 'little', signed=True) for s in frames)

    # Build WAV header
    file_size = 36 + len(data)
    riff = b'RIFF'
    size = struct.pack('<I', file_size - 8)
    wave = b'WAVE'
    fmt_chunk = b'fmt '
    fmt_size = struct.pack('<I', 16)
    audio_format = struct.pack('<H', 1)  # PCM
    channels = struct.pack('<H', 1)
    sample_rate_b = struct.pack('<I', sample_rate)
    byte_rate = struct.pack('<I', sample_rate * 1 * 2)  # sample_rate * channels * (bits/8)
    block_align = struct.pack('<H', 2)
    bits_per_sample = struct.pack('<H', 16)
    data_chunk = b'data'
    data_size = struct.pack('<I', len(data))

    # Concatenate header and data
    wav_data = riff + size + wave + fmt_chunk + fmt_size + audio_format + channels + \
               sample_rate_b + byte_rate + block_align + bits_per_sample + data_chunk + \
               data_size + data

    return wav_data


# Note: To actually create the kick.wav file, run this script as a standalone
# and uncomment the following lines:
if __name__ == '__main__':
    with open('kick.wav', 'wb') as f:
        f.write(generate_kick_wav())