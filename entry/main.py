"""
FastAPI application entry point for Kokoro TTS API
"""
import os
import sys
import asyncio
import threading
import uvicorn
from typing import Optional, Dict
from fastapi import FastAPI, Response, status, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

from entry.config import get_settings, is_container_environment, parse_bool_env
from entry.routers import tts, jobs, voices, streams, debug
from entry.core.models import (
    initialize_models, 
    get_voices, 
    get_models, 
    get_pipelines
)

# Global flags to track model loading state
MODELS_LOADED = False
initialized = False
initialization_error = None

# Add thread lock for synchronization
model_init_lock = threading.RLock()

async def check_initialization(request: Request, call_next):
    """Middleware to check if models are initialized before processing requests"""
    # Skip initialization check for health check and static files
    if request.url.path in ["/health", "/favicon.ico"] or request.url.path.startswith("/static/"):
        return await call_next(request)
    
    # Check if models are loaded
    models = get_models()
    voices = get_voices()
    
    if len(models) == 0 or len(voices) == 0:
        error_msg = f"Service not ready - models not initialized. Models: {len(models)}, Voices: {len(voices)}"
        if len(error_msg) > 200:
            error_msg = error_msg[:197] + "..."
        
        logger.error(error_msg)
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service not ready",
                "detail": "Models are still initializing. Please try again in a few moments.",
                "models_loaded": len(models),
                "voices_loaded": len(voices)
            }
        )
    
    logger.info(f"TTS middleware validation passed: models={len(models)}, voices={len(voices)}")
    return await call_next(request)

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    load_dotenv()
    settings = get_settings()
    
    app = FastAPI(
        title="Kokoro TTS API",
        description="REST API for Kokoro Text-to-Speech Engine",
        version="1.0.0",
        docs_url="/docs" if not settings.is_container else None,
        redoc_url="/redoc" if not settings.is_container else None
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add initialization check middleware
    app.middleware("http")(check_initialization)

    # Initialize models on startup
    @app.on_event("startup")
    async def startup_event():
        """Initialize models on startup"""
        global MODELS_LOADED, initialized, initialization_error
        
        logger.info("Initializing models - startup process beginning")
        
        # Start the queue processor for background job processing
        from entry.services.queue import start_queue_processor
        start_queue_processor()
        logger.info("Queue processor started for background job processing")
        
        # For Cloud Run, we need to make initialization non-blocking
        # to allow the health check endpoint to respond quickly
        def init_models_thread():
            global MODELS_LOADED, initialized, initialization_error
            try:
                # Use thread lock for safe initialization
                with model_init_lock:
                    # Determine if we need to force online mode for container initialization
                    force_online = False
                    if os.environ.get('CONTAINER_ENV', '').lower() == 'true' or \
                       os.environ.get('K_SERVICE', '').lower() != '':
                        force_online = not os.path.exists(os.path.join(os.getcwd(), 'models', 'Kokoro-82M'))
                        if force_online:
                            logger.info("Container environment detected, forcing online mode for first run")
                    
                    logger.info("Loading models in background thread")
                    # Force a complete model initialization synchronously within the thread
                    initialize_models(force_online=force_online)
                    
                    # Verify models are properly loaded by checking outputs of core functions
                    models = get_models()
                    pipelines = get_pipelines()
                    voices = get_voices()
                    
                    # Validate critical components are loaded
                    if not models or not pipelines or not voices:
                        raise RuntimeError(f"Critical components missing after initialization: models={bool(models)}, pipelines={bool(pipelines)}, voices={bool(voices)}")
                        
                    logger.info(f"Models loaded successfully: {len(models)} models, {len(pipelines)} pipelines, {len(voices)} voices")
                    MODELS_LOADED = True
                    initialized = True  # Set initialized flag when models are loaded
                    initialization_error = None  # Clear any previous errors
            except Exception as e:
                logger.error(f"Error loading models: {e}")
                import traceback
                logger.error(f"Initialization error traceback: {traceback.format_exc()}")
                initialization_error = str(e)  # Set error message
                MODELS_LOADED = False
                initialized = False
        
        # Start initialization in a background thread
        thread = threading.Thread(target=init_models_thread)
        thread.daemon = True
        thread.start()

    # Health check endpoints for Cloud Run
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Basic health check that always returns 200 OK (for initial container startup)"""
        if initialization_error:
            return {"status": "error", "error": initialization_error}
        if not initialized:
            return {"status": "initializing"}
        return {"status": "healthy", "voices": list(get_voices())}
    
    @app.get("/ready", tags=["Health"])
    async def readiness_check(response: Response):
        """Readiness check that returns 200 only when models are fully loaded"""
        if initialization_error:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "error", "error": initialization_error}
        if not initialized:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "loading", "message": "Models are still loading"}
        return {"status": "ready", "message": "Models loaded and ready"}
    
    # Include routers
    app.include_router(tts.router, prefix="/tts", tags=["TTS"])
    app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
    app.include_router(voices.router, prefix="/voices", tags=["Voices"])
    app.include_router(streams.router, prefix="/streams", tags=["Streams"])
    app.include_router(debug.router, prefix="/debug", tags=["Debug"])

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