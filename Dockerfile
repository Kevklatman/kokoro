# Use an official Python base image
FROM python:3.10-slim

# Set environment variables for production
ENV ENV=production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HOST=0.0.0.0
ENV RELOAD=False
ENV STREAMS_DIR=/app/streams
ENV OFFLINE_MODE=true

# Set work directory
WORKDIR /app

# Install system dependencies (add more as needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Create necessary directories
RUN mkdir -p ${STREAMS_DIR}

# Copy the project first for downloading models
COPY . /app/

# Make sure the scripts directory is executable
RUN chmod +x /app/scripts/*.py

# Create models directory
RUN mkdir -p /app/models

# Download models during build (set HF_TOKEN as build arg if needed)
ARG HF_TOKEN=
RUN if [ -n "$HF_TOKEN" ]; then \
    echo "Using provided Hugging Face token"; \
    HF_TOKEN=$HF_TOKEN MODELS_DIR=/app/models python3 /app/scripts/download_models.py; \
    else \
    echo "No Hugging Face token provided, using anonymous downloads"; \
    MODELS_DIR=/app/models python3 /app/scripts/download_models.py; \
    fi

# Expose the configured port
EXPOSE ${PORT}

# Use Gunicorn with Uvicorn workers for production
CMD gunicorn entry.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind ${HOST}:${PORT}