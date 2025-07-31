#!/usr/bin/env python3
"""
Debug script to test the document processing pipeline
"""
import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_document_processing():
    """Test the document processing pipeline step by step"""
    
    print("🔍 Testing Document Processing Pipeline...")
    print("=" * 50)
    
    # Test 1: Import modules
    print("1. Testing imports...")
    try:
        from src.document_processor import DocumentProcessor
        from src.llm_service import LLMService
        from src.vector_service import VectorService
        from src.query_processor import QueryProcessor
        print("✅ All modules imported successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Check environment variables
    print("\n2. Checking environment variables...")
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key and gemini_key != 'your_gemini_api_key_here':
        print("✅ Gemini API key found")
    else:
        print("❌ Gemini API key missing or invalid")
        return False
    
    # Test 3: Initialize services
    print("\n3. Initializing services...")
    try:
        document_processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        llm_service = LLMService()
        print("✅ LLM service initialized")
        
        vector_service = VectorService()
        print("✅ Vector service initialized")
        
        query_processor = QueryProcessor(document_processor, llm_service, vector_service)
        print("✅ Query processor initialized")
    except Exception as e:
        print(f"❌ Service initialization error: {e}")
        return False
    
    # Test 4: Test document download
    print("\n4. Testing document download...")
    test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    try:
        content = await document_processor.download_document(test_url)
        print(f"✅ Document downloaded successfully ({len(content)} bytes)")
    except Exception as e:
        print(f"❌ Document download error: {e}")
        return False
    
    # Test 5: Test text extraction
    print("\n5. Testing text extraction...")
    try:
        text = document_processor.extract_text_from_pdf(content)
        print(f"✅ Text extracted successfully ({len(text)} characters)")
        if len(text) < 100:
            print(f"⚠️  Warning: Text seems too short: '{text[:100]}...'")
    except Exception as e:
        print(f"❌ Text extraction error: {e}")
        return False
    
    # Test 6: Test document processing
    print("\n6. Testing full document processing...")
    try:
        document_data = await document_processor.process_document(test_url)
        print(f"✅ Document processed successfully:")
        print(f"   - Chunks: {document_data['total_chunks']}")
        print(f"   - Characters: {document_data['total_characters']}")
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return False
    
    # Test 7: Test LLM service
    print("\n7. Testing LLM service...")
    try:
        test_question = "What is this document about?"
        query_intent = await llm_service.parse_query_intent(test_question)
        print(f"✅ Query intent parsed: {query_intent}")
    except Exception as e:
        print(f"❌ LLM service error: {e}")
        return False
    
    # Test 8: Test embeddings
    print("\n8. Testing embeddings generation...")
    try:
        test_texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = await llm_service.generate_embeddings(test_texts)
        print(f"✅ Embeddings generated: {len(embeddings)} vectors of dimension {len(embeddings[0])}")
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        return False
    
    # Test 9: Test full query processing
    print("\n9. Testing full query processing...")
    try:
        test_questions = ["What is this document about?"]
        answers = await query_processor.process_queries(test_url, test_questions)
        print(f"✅ Query processed successfully:")
        for i, answer in enumerate(answers):
            print(f"   Q{i+1}: {answer[:100]}...")
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        return False
    
    print("\n🎉 All tests passed! The system is working correctly.")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_document_processing())
    sys.exit(0 if success else 1)
