#!/usr/bin/env python3
"""
Debug document processing to understand why accuracy is low
"""
import asyncio
import sys
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

async def debug_document_processing():
    """Debug the document processing pipeline"""
    
    print("🔍 Debugging Document Processing Pipeline...")
    print("=" * 60)
    
    # Test document
    test_url = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    try:
        # Import the functions from main.py
        sys.path.append('.')
        from main import (
            download_document, extract_text_from_pdf, clean_text, 
            chunk_text, generate_embeddings, find_relevant_chunks
        )
        
        print("✅ Successfully imported functions")
        
        # Step 1: Download document
        print("\n1. Downloading document...")
        content = await download_document(test_url)
        print(f"✅ Downloaded {len(content)} bytes")
        
        # Step 2: Extract text
        print("\n2. Extracting text from PDF...")
        raw_text = extract_text_from_pdf(content)
        print(f"✅ Extracted {len(raw_text)} characters")
        print(f"📝 First 500 chars: {raw_text[:500]}...")
        
        # Step 3: Clean text
        print("\n3. Cleaning text...")
        cleaned_text = clean_text(raw_text)
        print(f"✅ Cleaned text: {len(cleaned_text)} characters")
        print(f"📝 First 500 chars: {cleaned_text[:500]}...")
        
        # Step 4: Chunk text
        print("\n4. Chunking text...")
        chunks = chunk_text(cleaned_text)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Show sample chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n📄 Chunk {i+1} (length: {chunk['length']}):")
            print(f"   {chunk['text'][:200]}...")
        
        # Step 5: Search for specific terms
        print("\n5. Searching for key terms in text...")
        key_terms = [
            "waiting period", "pre-existing", "grace period", "premium", 
            "maternity", "sum insured", "room rent", "cataract", 
            "AYUSH", "exclusion", "organ donor", "policy term"
        ]
        
        found_terms = {}
        for term in key_terms:
            count = cleaned_text.lower().count(term.lower())
            found_terms[term] = count
            if count > 0:
                print(f"✅ Found '{term}': {count} times")
            else:
                print(f"❌ Missing '{term}'")
        
        # Step 6: Test embeddings
        print("\n6. Testing embeddings...")
        sample_texts = [chunks[0]['text'], chunks[1]['text']] if len(chunks) >= 2 else [chunks[0]['text']]
        embeddings = generate_embeddings(sample_texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"📊 Embedding dimensions: {len(embeddings[0])}")
        
        # Step 7: Test semantic search
        print("\n7. Testing semantic search...")
        test_question = "What is the waiting period for pre-existing diseases?"
        
        # Create mock document data
        document_data = {
            'chunks': chunks,
            'embeddings': embeddings
        }
        
        # Add embeddings to chunks for testing
        chunk_texts = [chunk['text'] for chunk in chunks]
        all_embeddings = generate_embeddings(chunk_texts)
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = all_embeddings[i]
        
        relevant_chunks = find_relevant_chunks(test_question, document_data, top_k=5)
        print(f"✅ Found {len(relevant_chunks)} relevant chunks")
        
        for i, chunk in enumerate(relevant_chunks):
            similarity = chunk.get('similarity_score', 0)
            print(f"\n🔍 Relevant chunk {i+1} (similarity: {similarity:.3f}):")
            print(f"   {chunk['text'][:300]}...")
        
        # Step 8: Manual search for waiting period info
        print("\n8. Manual search for 'waiting period' information...")
        waiting_period_chunks = []
        for chunk in chunks:
            if 'waiting period' in chunk['text'].lower():
                waiting_period_chunks.append(chunk)
        
        print(f"✅ Found {len(waiting_period_chunks)} chunks containing 'waiting period'")
        for i, chunk in enumerate(waiting_period_chunks[:3]):
            print(f"\n📋 Waiting period chunk {i+1}:")
            print(f"   {chunk['text'][:400]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in debugging: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_document_processing())
    if success:
        print("\n🎉 Debugging completed successfully!")
    else:
        print("\n💥 Debugging failed!")
