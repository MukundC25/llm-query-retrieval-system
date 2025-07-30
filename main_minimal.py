"""
Minimal FastAPI app for Vercel testing
"""
from fastapi import FastAPI
import os
import time

app = FastAPI(title="LLM Query System - Minimal")

@app.get("/")
async def root():
    """Basic health check"""
    return {
        "status": "healthy",
        "message": "Minimal version working",
        "timestamp": time.time(),
        "environment": os.getenv("VERCEL", "local")
    }

@app.get("/test")
async def test():
    """Test endpoint"""
    return {"test": "working"}

# Vercel handler
handler = app
