"""
Convenience entry point to run the TRUSTLENS Deepfake Detection API server.
"""

import sys
import os
from pathlib import Path

# Configure utf-8 encoding for Windows console compatibility
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from app.config import APP_NAME

if __name__ == "__main__":
    print("=================================================================")
    print(f"[TRUSTLENS] {APP_NAME}")
    print("API Endpoint: http://localhost:8001")
    print("Swagger Docs: http://localhost:8001/docs")
    print("Health Check: http://localhost:8001/api/media/health")
    print("=================================================================\n")

    port = int(os.getenv("PORT", 8001))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
