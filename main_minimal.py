#!/usr/bin/env python3
"""
Minimal working version for HackRX 2025 - Focus on core accuracy
"""
import os
import logging
import time
import requests
import io
import re
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini AI configured successfully")
else:
    logger.error("GEMINI_API_KEY not found in environment variables")

# FastAPI app
app = FastAPI(
    title="LLM Query-Retrieval System",
    description="Minimal working system for HackRX 2025",
    version="1.0.0"
)

# Security
security = HTTPBearer()
BEARER_TOKEN = "12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Data models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Simple document cache
document_cache = {}

def get_gemini_model():
    """Get Gemini model"""
    try:
        return genai.GenerativeModel('gemini-pro')
    except Exception as e:
        logger.error(f"Error getting Gemini model: {e}")
        return None

async def download_document(url: str) -> bytes:
    """Download document from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Error downloading document: {e}")
        raise Exception(f"Failed to download document: {e}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF"""
    try:
        laparams = LAParams(
            boxes_flow=0.5,
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5
        )
        text = extract_text(io.BytesIO(pdf_content), laparams=laparams)
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        raise Exception(f"Failed to extract PDF text: {e}")

def clean_text(text: str) -> str:
    """Clean extracted text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks"""
    chunks = []
    words = text.split()
    
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def find_relevant_chunks(question: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Find relevant chunks using simple keyword matching"""
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        chunk_words = set(chunk_lower.split())
        
        # Simple scoring based on word overlap
        overlap = len(question_words.intersection(chunk_words))
        
        # Bonus for insurance keywords
        insurance_keywords = ['policy', 'coverage', 'benefit', 'premium', 'insured', 'claim']
        insurance_score = sum(1 for keyword in insurance_keywords if keyword in chunk_lower)
        
        total_score = overlap + insurance_score * 0.5
        
        if total_score > 0:
            scored_chunks.append((chunk, total_score))
    
    # Sort by score and return top chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k]]

async def answer_question(question: str, relevant_chunks: List[str]) -> str:
    """Generate answer using Gemini"""
    try:
        model = get_gemini_model()
        if model is None:
            return "Error: AI service is not available"
        
        if not relevant_chunks:
            return "This information is not available in the provided document"
        
        # Prepare context
        context = "\n\n".join([f"Section {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks)])
        
        # Simple but effective prompt
        prompt = f"""Based on the following document sections, answer the question accurately.

DOCUMENT SECTIONS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based only on the information in the document sections above
- Be specific and include exact details when available
- If the information is not in the document, say "This information is not available in the provided document"

ANSWER:"""

        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "Error: No response generated"
            
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error: Unable to generate answer - {str(e)}"

async def process_document_and_questions(document_url: str, questions: List[str]) -> List[str]:
    """Process document and answer questions"""
    try:
        # Check cache
        if document_url in document_cache:
            chunks = document_cache[document_url]
        else:
            # Download and process document
            content = await download_document(document_url)
            text = extract_text_from_pdf(content)
            cleaned_text = clean_text(text)
            chunks = chunk_text(cleaned_text)
            document_cache[document_url] = chunks
        
        # Answer questions
        answers = []
        for question in questions:
            relevant_chunks = find_relevant_chunks(question, chunks)
            answer = await answer_question(question, relevant_chunks)
            answers.append(answer)
        
        return answers
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return [f"Error: Document processing failed - {str(e)}"] * len(questions)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Minimal LLM Query Retrieval System is running",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint"""
    return """
    <html>
        <head><title>LLM Query-Retrieval System</title></head>
        <body>
            <h1>LLM Query-Retrieval System</h1>
            <p>Minimal working version for HackRX 2025</p>
            <p>Use POST /api/v1/hackrx/run for document queries</p>
        </body>
    </html>
    """

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def hackrx_document_processing(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Main endpoint for processing document queries"""
    try:
        start_time = time.time()
        
        # Process document and questions
        answers = await process_document_and_questions(request.documents, request.questions)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(request.questions)} questions in {processing_time:.2f} seconds")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in hackrx_document_processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
