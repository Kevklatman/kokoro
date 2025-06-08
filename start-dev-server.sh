#!/bin/bash

# Start the development server for Kokoro API
echo "Starting Kokoro API development server..."
echo "This server will accept connections from your Swift frontend"

# Get the local IP address
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
echo "Your local IP address is: $LOCAL_IP"
echo "Your Swift app should connect to: http://$LOCAL_IP:8080"

# Start the Docker container
docker-compose -f docker-compose.dev.yml up --build
