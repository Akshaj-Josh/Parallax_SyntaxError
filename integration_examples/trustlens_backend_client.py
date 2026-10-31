"""
TRUSTLENS Backend Integration Client.

Use this client in your existing Python backend to call the Deepfake Detection Module,
or copy the code into your existing pipeline.
"""

import requests
from typing import Dict, Any, Optional


class TrustlensDeepfakeClient:
    """
    Python client for communicating with the TRUSTLENS Deepfake Detection API.
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> Dict[str, Any]:
        """Checks if the deepfake detection service is reachable."""
        url = f"{self.base_url}/api/media/health"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        return res.json()

    def analyze_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Uploads and analyzes an audio file for deepfake/synthetic voice indicators.
        Returns the structured MediaAnalysisResponse.
        """
        url = f"{self.base_url}/api/media/analyze-audio"
        with open(audio_file_path, "rb") as f:
            files = {"file": (audio_file_path, f, "audio/wav")}
            res = requests.post(url, files=files, timeout=60)

        if not res.ok:
            raise RuntimeError(f"Deepfake detection failed ({res.status_code}): {res.text}")

        return res.json()

    def analyze_video(self, video_file_path: str) -> Dict[str, Any]:
        """
        Uploads and analyzes a video file for facial manipulation.
        """
        url = f"{self.base_url}/api/media/analyze-video"
        with open(video_file_path, "rb") as f:
            files = {"file": (video_file_path, f, "video/mp4")}
            res = requests.post(url, files=files, timeout=90)

        if not res.ok:
            raise RuntimeError(f"Video detection failed ({res.status_code}): {res.text}")

        return res.json()


# -----------------------------------------------------------------------------
# Direct FastAPI Router Integration Example (No HTTP microservice overhead):
# -----------------------------------------------------------------------------
"""
If your existing backend is built with FastAPI, you don't even need to run a
separate server! You can mount the router directly into your existing app:

    from fastapi import FastAPI
    from app.routers.media import router as deepfake_router

    app = FastAPI()

    # Mount the deepfake detection endpoints directly
    app.include_router(deepfake_router)
"""

if __name__ == "__main__":
    # Quick test example
    client = TrustlensDeepfakeClient("http://localhost:8001")
    try:
        health = client.check_health()
        print("Deepfake Service Status:", health)
    except Exception as e:
        print("Deepfake Service Not Running yet:", e)
