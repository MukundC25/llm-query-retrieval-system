#!/usr/bin/env python3
"""
Quick test script to verify Vercel deployment compatibility
"""

import sys
import os

# Add src to path
sys.path.append('src')

def test_imports():
    """Test that all required modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test core dependencies
        import fastapi
        print("✅ FastAPI imported")
        
        import uvicorn
        print("✅ Uvicorn imported")
        
        import google.generativeai as genai
        print("✅ Google Generative AI imported")
        
        # Test our modules
        from src.llm_service import LLMService
        print("✅ LLM Service imported")
        
        from src.vector_service import VectorService
        print("✅ Vector Service imported")
        
        from src.document_processor import DocumentProcessor
        print("✅ Document Processor imported")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_llm_service():
    """Test LLM service initialization"""
    try:
        print("\nTesting LLM Service...")
        
        # Set dummy API key for testing
        os.environ['GEMINI_API_KEY'] = 'test-key'
        
        from src.llm_service import LLMService
        llm = LLMService()
        
        # Test fallback embeddings
        texts = ["Hello world", "Test document"]
        embeddings = llm._generate_simple_embeddings(texts)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"✅ Embedding dimension: {len(embeddings[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Service error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Vercel Compatibility Test\n")
    
    success = True
    success &= test_imports()
    success &= test_llm_service()
    
    if success:
        print("\n🎉 All tests passed! Ready for Vercel deployment.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check dependencies.")
        sys.exit(1)
