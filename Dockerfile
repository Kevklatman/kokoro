# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
# Set environment variables for production
ENV ENV=production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (add more as needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Copy your code
COPY . .

# Expose port (Cloud Run uses 8080 by default)
EXPOSE 8080

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "entry.api:app", "--host", "127.0.0.1", "--port", "8080"]
