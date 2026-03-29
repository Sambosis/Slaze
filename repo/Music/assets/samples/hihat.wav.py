import struct
import wave
import random

# Hi-hat sample: short white noise burst
# 16-bit mono WAV at 44.1kHz, duration ~200ms
sample_rate = 44100
duration = 0.2
num_samples = int(sample_rate * duration)

# Generate white noise samples
samples = []
for _ in range(num_samples):
    # Scale random value to 16-bit signed integer range, reduce amplitude for less harshness
    sample = int(random.uniform(-32768 * 0.7, 32767 * 0.7))
    # Little-endian packing
    samples.append(struct.pack('<h', sample))

# Write to hihat.wav
wave_file = wave.open('hihat.wav', 'wb')
wave_file.setnchannels(1)  # Mono
wave_file.setsampwidth(2)  # 16-bit
wave_file.setframerate(sample_rate)
wave_file.writeframes(b''.join(samples))
wave_file.close()