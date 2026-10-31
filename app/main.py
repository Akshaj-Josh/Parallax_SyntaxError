"""
TRUSTLENS Deepfake Detection Backend Entrypoint.
Exposes FastAPI application with CORS and security middlewares.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import APP_NAME, APP_VERSION, CORS_ORIGINS
from app.routers.media import router as media_router

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "TRUSTLENS Deepfake & Synthetic Media Detection Module. "
        "Provides genuine pretrained audio deepfake detection, physical acoustic "
        "artifact analysis, ephemeral privacy guarantees, and safe verification guidance."
    ),
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for integration with existing frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Register Media Analysis Router
app.include_router(media_router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "GET /api/media/health",
            "analyze_audio": "POST /api/media/analyze-audio",
            "analyze_video": "POST /api/media/analyze-video"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "message": f"Internal server error: {str(exc)}",
            "endpoint": request.url.path
        }
    )
