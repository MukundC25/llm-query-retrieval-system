"""
Vercel API handler for the LLM Query-Retrieval System
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# Vercel handler
def handler(request, response):
    return app(request, response)

# Export the app for Vercel
app = app
