"""
TRUSTLENS Test Sample Generator (Zero-Dependency Python Standard Library).
Generates safe synthetic and authentic-baseline test WAV files
using pure Python standard library ('wave', 'struct', 'math', 'random').
License: CC0 1.0 Universal (Public Domain).
"""

import os
import wave
import struct
import math
import random
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "test_data"


def save_pcm_wav(file_path: Path, samples: list, sample_rate: int = 16000):
    """Saves float samples in [-1.0, 1.0] to 16-bit mono PCM WAV."""
    os.makedirs(file_path.parent, exist_ok=True)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        # Pack into signed 16-bit integers
        packed_frames = bytearray()
        for s in samples:
            clamped = max(-0.999, min(0.999, s))
            int_val = int(clamped * 32767.0)
            packed_frames.extend(struct.pack("<h", int_val))

        wf.writeframes(packed_frames)
    print(f"  [+] Generated: {file_path.name} ({len(samples)/sample_rate:.2f}s, {file_path.stat().st_size} bytes)")


def create_all_test_samples():
    print(f"[TRUSTLENS] Generating safe test audio samples in {OUTPUT_DIR}...")
    sample_rate = 16000

    # -------------------------------------------------------------
    # 1. Authentic Speech Simulation (Biological jitter, room floor, wideband)
    # -------------------------------------------------------------
    duration = 4.0
    total_samples = int(sample_rate * duration)
    samples_auth = []

    f0_base = 135.0
    phase = 0.0

    for i in range(total_samples):
        t = i / float(sample_rate)
        # Natural intonation drift
        f0 = f0_base + 12.0 * math.sin(2 * math.pi * 0.8 * t)
        # 1.2% biological vocal fold micro-jitter
        jitter = random.gauss(0, 1.5)
        phase += 2 * math.pi * (f0 + jitter) / sample_rate

        # Natural acoustic formant harmonics
        voice = (
            0.50 * math.sin(phase) +
            0.30 * math.sin(2 * phase) +
            0.18 * math.sin(3 * phase) +
            0.10 * math.sin(4 * phase) +
            0.05 * math.sin(5 * phase)
        )

        # Ambient microphone/room noise floor
        room_noise = random.gauss(0, 0.008)
        samples_auth.append(voice * 0.75 + room_noise)

    save_pcm_wav(OUTPUT_DIR / "sample_authentic_voice.wav", samples_auth, sample_rate)

    # -------------------------------------------------------------
    # 2. Synthetic Speech Simulation (TTS / Vocoder artifacts)
    # - Strict harmonic stack without micro-jitter (robotic pitch)
    # - Artificial zero-energy pause (digital silence)
    # -------------------------------------------------------------
    duration = 4.0
    total_samples = int(sample_rate * duration)
    samples_synth = []
    f0_synth = 145.0

    for i in range(total_samples):
        t = i / float(sample_rate)
        phase_synth = 2 * math.pi * f0_synth * t

        # Digital pause window (1.5s to 2.2s) - pure mathematical zero
        if 1.5 <= t <= 2.2:
            samples_synth.append(0.0)
        else:
            voice_synth = (
                0.60 * math.sin(phase_synth) +
                0.35 * math.sin(2 * phase_synth) +
                0.15 * math.sin(3 * phase_synth)
            )
            samples_synth.append(voice_synth * 0.80)

    save_pcm_wav(OUTPUT_DIR / "sample_synthetic_deepfake.wav", samples_synth, sample_rate)

    # -------------------------------------------------------------
    # 3. Invalid Short Sample (0.2s duration - should be rejected by validator)
    # -------------------------------------------------------------
    short_samples = [
        0.5 * math.sin(2 * math.pi * 200.0 * (i / sample_rate))
        for i in range(int(sample_rate * 0.2))
    ]
    save_pcm_wav(OUTPUT_DIR / "sample_too_short_invalid.wav", short_samples, sample_rate)

    print("[TRUSTLENS] All test samples generated successfully.\n")


if __name__ == "__main__":
    create_all_test_samples()
