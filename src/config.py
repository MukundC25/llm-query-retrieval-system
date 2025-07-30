"""
Configuration module for the LLM Query-Retrieval System
"""

import os
from typing import Optional

class Config:
    """Application configuration"""
    
    # API Configuration
    API_TITLE = "LLM Query-Retrieval System"
    API_DESCRIPTION = "AI system for processing documents and answering natural language queries"
    API_VERSION = "1.0.0"
    
    # Authentication
    BEARER_TOKEN = "12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f"
    
    # Gemini Configuration
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', 'AIzaSyAN0Upn-E8l1m9KL99p6Mrq8fe-DTfgpOM')
    GEMINI_MODEL = "gemini-pro"
    GEMINI_MAX_TOKENS = 4000
    GEMINI_TEMPERATURE = 0.1
    
    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT: str = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp-free')
    PINECONE_INDEX_NAME: str = os.getenv('PINECONE_INDEX_NAME', 'document-embeddings')
    PINECONE_DIMENSION = 384  # Updated for sentence-transformers model
    
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Query Processing
    MAX_QUESTIONS_PER_REQUEST = 20
    VECTOR_SEARCH_TOP_K = 5
    MAX_CONTEXT_TOKENS = 2500
    
    # Performance
    REQUEST_TIMEOUT = 30  # seconds
    DOWNLOAD_TIMEOUT = 30  # seconds
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.GEMINI_API_KEY:
            return False
        return True

    @classmethod
    def get_summary(cls) -> dict:
        """Get configuration summary (without sensitive data)"""
        return {
            "api_version": cls.API_VERSION,
            "gemini_model": cls.GEMINI_MODEL,
            "chunk_size": cls.CHUNK_SIZE,
            "max_questions": cls.MAX_QUESTIONS_PER_REQUEST,
            "vector_top_k": cls.VECTOR_SEARCH_TOP_K,
            "has_gemini_key": bool(cls.GEMINI_API_KEY),
            "has_pinecone_key": bool(cls.PINECONE_API_KEY),
            "pinecone_environment": cls.PINECONE_ENVIRONMENT,
            "pinecone_index": cls.PINECONE_INDEX_NAME
        }
