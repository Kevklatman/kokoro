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
ENV OFFLINE_MODE=false
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt

# Create necessary directories
RUN mkdir -p ${STREAMS_DIR} \
    && mkdir -p /app/models/Kokoro-82M \
    && mkdir -p /app/models/voices \
    && mkdir -p /tmp/models_tmp

# Copy the model checker script
COPY check_models.py /app/

# Create a startup script for model checking and server startup
RUN echo '#!/bin/bash' > /app/start.sh \
    && echo 'echo "Checking model files..."' >> /app/start.sh \
    && echo 'python /app/check_models.py' >> /app/start.sh \
    && echo '' >> /app/start.sh \
    && echo 'if [ "$OFFLINE_MODE" != "true" ]; then' >> /app/start.sh \
    && echo '  echo "Online mode enabled, will download models if needed"' >> /app/start.sh \
    && echo 'fi' >> /app/start.sh \
    && echo '' >> /app/start.sh \
    && echo 'exec gunicorn entry.main:app --bind ${HOST}:${PORT} --workers 1 --timeout 0 "$@"' >> /app/start.sh \
    && chmod +x /app/start.sh

# First, copy the entire project to handle model files
COPY . /tmp/project/

# Now handle models safely without risky wildcards
RUN if [ -f "/tmp/project/models/config.json" ]; then \
      cp /tmp/project/models/config.json /app/models/config.json && \
      cp /tmp/project/models/config.json /app/models/Kokoro-82M/config.json && \
      echo "Config.json copied" || echo "Failed to copy config.json"; \
    else \
      echo "No config.json found"; \
    fi

# Handle model file
RUN if [ -f "/tmp/project/models/kokoro-v1_0.pth" ]; then \
      cp /tmp/project/models/kokoro-v1_0.pth /app/models/Kokoro-82M/kokoro-v1_0.pth && \
      echo "Model file copied" || echo "Failed to copy model file"; \
    else \
      echo "No kokoro-v1_0.pth found"; \
    fi

# Handle voice files
RUN if [ -d "/tmp/project/models/voices" ]; then \
      find /tmp/project/models/voices -type f -name "*.pt" -exec cp {} /app/models/voices/ \; && \
      echo "Voice files copied" || echo "No voice files found or failed to copy"; \
    else \
      echo "No voices directory found"; \
    fi

# Copy the project code after handling models
COPY . /app/

# Set proper permissions for all files
RUN chmod -R 755 /app

# Expose the port
EXPOSE ${PORT}

# Use our startup script
ENTRYPOINT ["/app/start.sh"]