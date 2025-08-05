#!/usr/bin/env python3
"""
Ultra-minimal working system for HackRX 2025 - Phase 1 Recovery
"""
import os
import logging
import time
import requests
import re
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai

# Try PyMuPDF first, fallback to pdfminer
try:
    import fitz  # PyMuPDF
    PDF_LIBRARY = "pymupdf"
except ImportError:
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        import io
        PDF_LIBRARY = "pdfminer"
    except ImportError:
        PDF_LIBRARY = "none"

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

# Templates configuration
templates = Jinja2Templates(directory="templates")

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
    """Get Gemini model - try available models"""
    try:
        # Try the latest available models
        for model_name in ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro']:
            try:
                model = genai.GenerativeModel(model_name)
                logger.info(f"Successfully loaded model: {model_name}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue

        logger.error("No Gemini models available")
        return None
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
    """Extract text from PDF using best available method"""
    try:
        if PDF_LIBRARY == "pymupdf":
            # Use PyMuPDF (faster and more reliable)
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            logger.info(f"Extracted {len(text)} characters using PyMuPDF")
            return text

        elif PDF_LIBRARY == "pdfminer":
            # Fallback to pdfminer
            laparams = LAParams(
                boxes_flow=0.5,
                word_margin=0.1,
                char_margin=2.0,
                line_margin=0.5
            )
            text = extract_text(io.BytesIO(pdf_content), laparams=laparams)
            logger.info(f"Extracted {len(text)} characters using pdfminer")
            return text
        else:
            raise Exception("No PDF extraction library available")

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
        
        # Enhanced prompt for better accuracy
        prompt = f"""You are a legal document assistant. Answer based only on this policy document.

POLICY DOCUMENT TEXT:
{context}

QUESTION: {question}

CRITICAL INSTRUCTIONS:
1. Read the document text carefully
2. Answer ONLY based on information explicitly stated in the document
3. Include specific details: numbers, percentages, time periods, conditions
4. Quote relevant text when providing specific information
5. If the exact information is not found, say "This information is not available in the provided document"
6. Be precise and factual - do not make assumptions

For insurance questions, look for:
- Waiting periods (pre-existing diseases, specific conditions)
- Grace periods (premium payments)
- Coverage amounts and limits
- Exclusions and restrictions
- Benefits and eligibility criteria

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

@app.get("/ping")
async def ping():
    """Simple ping endpoint for testing"""
    return {"status": "ok", "message": "pong"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Ultra-minimal LLM Query Retrieval System - Phase 1",
        "timestamp": time.time(),
        "version": "1.0.0",
        "pdf_library": PDF_LIBRARY,
        "gemini_configured": GEMINI_API_KEY is not None
    }

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint serving the original frontend template"""
    return templates.TemplateResponse("index.html", {"request": request})


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
