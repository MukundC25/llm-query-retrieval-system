#!/bin/bash
PORT=${PORT:-8000}
echo "Starting server on port $PORT"
echo "Environment variables:"
env | grep -E "(PORT|GEMINI|PINECONE|BEARER)" || echo "No relevant env vars found"
exec uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info
