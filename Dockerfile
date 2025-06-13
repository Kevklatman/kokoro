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

# Create models directory structure
RUN mkdir -p /app/models/Kokoro-82M /app/models/voices

# Copy model files first
COPY models/config.json /app/models/config.json
COPY models/config.json /app/models/Kokoro-82M/config.json
COPY models/kokoro-v1_0.pth /app/models/Kokoro-82M/kokoro-v1_0.pth
COPY models/voices/*.pt /app/models/voices/

# Copy the rest of the project
COPY . /app/

# Expose the configured port
EXPOSE ${PORT}

# Use Gunicorn with Uvicorn workers for production
CMD gunicorn entry.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind ${HOST}:${PORT}