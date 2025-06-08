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


app = create_app()

if __name__ == "__main__":
    # Create streams directory if it doesn't exist
    os.makedirs("streams", exist_ok=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)