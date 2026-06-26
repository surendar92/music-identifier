# EE200 Course Project: Sonic Signatures & Signals to Softwares

An end-to-end audio fingerprinting system built for the **EE200: Signals, Systems and Networks** course project at **IIT Kanpur**. This application implements a robust audio identification pipeline—inspired by commercial systems like Shazam—capable of recognizing an unknown query clip by comparing its compact spectral signature against an indexed database of reference tracks.

---

##  Features

- **Single-Clip Mode:** Upload an audio clip to instantly identify the track and visualize the step-by-step pipeline outputs (Spectrogram, Constellation Peaks, and Offset Alignment Histogram).
- **Batch Evaluation Mode:** Upload multiple query clips simultaneously to automatically execute a parallelized search and download a structured `results.csv` sheet.
- **High Noise Immunity:** Uses anchor-target peak pairing to reliably match signals even when heavily corrupted by Gaussian white noise.
- **Pre-Indexed Database:** Ships with an active, high-performance SQLite database tracking nearly a million audio hashes across 50 reference tracks.

---

##  System Architecture & Pipeline

```text
[Audio Input] ──► [STFT Spectrogram] ──► [Constellation Map] ──► [Pairwise Hashes] ──► [Histogram Matching]

##  Repository Layout

```text
├── .idea/                 # IDE configurations
├── songs/                 # Directory holding reference audio files
├── app.py                 # Interactive Streamlit deployment script
├── build_database.py      # SQLite indexer for reference song parsing
├── fingerprint.py         # Modular signal processing functions (STFT, peaks, hashes)
├── music_database.db      # Pre-generated SQLite index containing processed hashes
├── requirements.txt       # System-level Python dependencies
└── song_metadata.json     # Auxiliary tracking data for indexed tracks

