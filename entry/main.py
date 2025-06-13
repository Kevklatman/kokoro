"""
FastAPI Application Entry Point
"""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from entry.config import get_settings
from entry.routers import tts, jobs, voices, streams
from entry.core.models import initialize_models


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    load_dotenv()
    settings = get_settings()
    
    app = FastAPI(
        title="Kokoro TTS API",
        description="REST API for Kokoro Text-to-Speech Engine",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize models on startup
    @app.on_event("startup")
    async def startup_event():
        initialize_models()

    # Include routers
    app.include_router(tts.router, prefix="/tts", tags=["TTS"])
    app.include_router(jobs.router, prefix="/tts", tags=["Jobs"])
    app.include_router(voices.router, tags=["Voices"])
    app.include_router(streams.router, prefix="/streams", tags=["Streaming"])

    @app.get("/")
    async def root():
        return {"message": "Kokoro TTS API is running. Visit /docs for API documentation."}

    return app


# Create the application instance
app = create_app()

# Always ensure streams directory exists
streams_dir = os.getenv("STREAMS_DIR", "streams")
os.makedirs(streams_dir, exist_ok=True)

# Only used when running this file directly (development mode)
if __name__ == "__main__":
    # Get configuration from environment variables
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    reload_mode = os.getenv("RELOAD", "False").lower() in ("true", "1", "t")
    
    print(f"Starting development server at {host}:{port} (reload={reload_mode})")
    uvicorn.run("entry.main:app", host=host, port=port, reload=reload_mode)
    
    # NOTE: For production, this block will not be executed.
    # Instead, the container orchestration should:
    # 1. Import the 'app' object directly
    # 2. Set up proper process management
    # 3. Configure via environment variables