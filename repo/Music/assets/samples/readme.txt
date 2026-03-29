Default Drum Samples
This folder must contain the four bundled WAV files the drum machine loads at startup so it can produce sound immediately:

- kick.wav (kick drum)
- snare.wav (snare drum)
- hihat.wav (hi-hat)
- clap.wav (hand clap)

Keep these filenames intact so the application finds them automatically. If any file is missing or unreadable, the app will launch but the corresponding track will remain silent until you provide a valid WAV.

Using Custom Samples
If you want different sounds, either replace the files above with new WAV assets that use the same names, or launch the app and use each track's sample selector to browse to another WAV file. Custom selections are stored inside presets, so saving a preset will remember both the pattern and the chosen samples.

Technical Notes
For best results use uncompressed PCM WAV files (44.1 kHz, 16-bit, mono or stereo). Samples placed here are treated as the defaults every time the drum machine starts.