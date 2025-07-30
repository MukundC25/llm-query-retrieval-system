#!/usr/bin/env python3
"""
Simple runner script for Railway deployment
"""
import os
import sys

def main():
    # Set PORT from environment
    port = int(os.getenv("PORT", 8000))
    
    # Import and run the main application
    try:
        import uvicorn
        from main import app
        
        print(f"Starting server on port {port}")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info"
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
