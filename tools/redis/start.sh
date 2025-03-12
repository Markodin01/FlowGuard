#!/bin/bash

# Check if Redis is running
REDIS_PID=$(pgrep redis-server)

if [ -n "$REDIS_PID" ]; then
    echo "Redis is running with PID: $REDIS_PID. Shutting down..."
    # Attempt to shutdown Redis gracefully
    redis-cli shutdown
    # Wait a bit to ensure Redis has stopped
    sleep 2
fi

echo "Starting Redis..."
# Start Redis server
redis-server

echo "Redis started successfully."
