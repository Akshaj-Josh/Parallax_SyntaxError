"""
Tests for API endpoint routing, file validation, and error handlers.
"""

import unittest
import os
import io
import wave
import struct
import math
from pathlib import Path

from app.utils.file_security import (
    validate_file_metadata,
    managed_temp_file,
    HTTPException,
    UploadFile
)

try:
    from app.main import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    app = None


class TestApiValidation(unittest.TestCase):
    def setUp(self):
        self.temp_wav = Path(__file__).resolve().parent / "test_temp.wav"
        # Create a simple 1-second WAV using pure standard library
        sample_rate = 16000
        with wave.open(str(self.temp_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            packed = bytearray()
            for i in range(sample_rate):
                s = 0.5 * math.sin(2 * math.pi * 440.0 * (i / sample_rate))
                packed.extend(struct.pack("<h", int(s * 32767.0)))
            wf.writeframes(packed)

    def tearDown(self):
        if self.temp_wav.exists():
            os.remove(self.temp_wav)

    def test_reject_unsupported_extensions(self):
        fake_file = UploadFile(filename="malicious.exe", file=io.BytesIO(b"dummy"))
        with self.assertRaises(HTTPException) as ctx:
            validate_file_metadata(fake_file, media_category="audio")
        self.assertEqual(ctx.exception.status_code, 415)

    def test_accept_valid_audio_extensions(self):
        for valid_ext in [".wav", ".mp3", ".m4a", ".flac"]:
            f = UploadFile(filename=f"audio{valid_ext}", file=io.BytesIO(b"data"))
            ext = validate_file_metadata(f, media_category="audio")
            self.assertEqual(ext, valid_ext)

    def test_app_route_registration_if_fastapi_installed(self):
        if HAS_FASTAPI and app is not None:
            routes = [r.path for r in app.routes]
            self.assertIn("/api/media/health", routes)
            self.assertIn("/api/media/analyze-audio", routes)
            self.assertIn("/api/media/analyze-video", routes)


if __name__ == "__main__":
    unittest.main()
