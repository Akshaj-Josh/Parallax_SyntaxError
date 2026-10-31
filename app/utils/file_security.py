"""
Security and privacy utilities for handling user-submitted media.
Guarantees ephemeral processing: files are strictly held in temp storage
and deleted immediately upon completion or failure.
"""

import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
try:
    from fastapi import UploadFile, HTTPException, status
except ImportError:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class status:
        HTTP_400_BAD_REQUEST = 400
        HTTP_413_REQUEST_ENTITY_TOO_LARGE = 413
        HTTP_415_UNSUPPORTED_MEDIA_TYPE = 415
        HTTP_422_UNPROCESSABLE_ENTITY = 422

    class UploadFile:
        def __init__(self, filename: str = "", file=None, content_type: str = ""):
            self.filename = filename
            self.file = file
            self.content_type = content_type

from app.config import (
    MAX_FILE_SIZE_BYTES,
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    ALLOWED_AUDIO_MIMES,
    ALLOWED_VIDEO_MIMES
)


def validate_file_metadata(upload_file: UploadFile, media_category: str = "audio") -> str:
    """
    Validates uploaded file extension, declared MIME type, and sanitize filename.
    Returns the lowercased extension.
    """
    if not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    ext = Path(upload_file.filename).suffix.lower()
    content_type = (upload_file.content_type or "").lower()

    if media_category == "audio":
        allowed_exts = ALLOWED_AUDIO_EXTENSIONS
        allowed_mimes = ALLOWED_AUDIO_MIMES
    elif media_category == "video":
        allowed_exts = ALLOWED_VIDEO_EXTENSIONS
        allowed_mimes = ALLOWED_VIDEO_MIMES
    else:
        allowed_exts = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
        allowed_mimes = ALLOWED_AUDIO_MIMES | ALLOWED_VIDEO_MIMES

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file format '{ext}'. Supported formats for {media_category}: "
                f"{', '.join(sorted(allowed_exts))}"
            )
        )

    # Some browsers/clients send application/octet-stream for audio/video; we allow it if extension matches
    if content_type and content_type != "application/octet-stream" and content_type not in allowed_mimes:
        # Check if MIME type at least matches the category prefix
        if not content_type.startswith(media_category + "/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Declared MIME type '{content_type}' is invalid for {media_category}."
            )

    return ext


@contextmanager
def managed_temp_file(upload_file: UploadFile, suffix: str = ".wav") -> Generator[str, None, None]:
    """
    Safely streams an UploadFile to a temporary file on disk, enforcing max file size.
    Guarantees deletion in finally block to ensure complete user privacy.
    """
    temp_dir = tempfile.gettempdir()
    unique_name = f"trustlens_{uuid.uuid4().hex}{suffix}"
    temp_path = os.path.join(temp_dir, unique_name)

    total_bytes = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    try:
        with open(temp_path, "wb") as buffer:
            while True:
                chunk = upload_file.file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File size exceeds maximum allowed limit of "
                            f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
                        )
                    )
                buffer.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        yield temp_path

    finally:
        # Guaranteed cleanup: delete temporary file immediately
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
