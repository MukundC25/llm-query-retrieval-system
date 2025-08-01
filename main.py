"""
Simplified LLM-Powered Intelligent Query-Retrieval System
Main FastAPI application - streamlined for hackathon submission
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Optional
import os
import logging
import time
import asyncio
import requests
import io
import re
from dotenv import load_dotenv

# Document processing imports
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from docx import Document

# AI imports
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Templates
templates = Jinja2Templates(directory="templates")

app = FastAPI(
    title="LLM Query-Retrieval System",
    description="AI system for processing documents and answering natural language queries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Expected bearer token
EXPECTED_TOKEN = "12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token authentication"""
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Request/Response models
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        if not v:
            raise ValueError('At least one question is required')
        if len(v) > 20:
            raise ValueError('Maximum 20 questions allowed per request')
        return v

class QueryResponse(BaseModel):
    answers: List[str]

# Initialize Gemini
def initialize_gemini():
    """Initialize Gemini AI"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable is required")
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        return None

# Global model instance
gemini_model = None

def get_gemini_model():
    """Get or initialize Gemini model"""
    global gemini_model
    if gemini_model is None:
        gemini_model = initialize_gemini()
    return gemini_model

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Gemini's embedding API"""
    try:
        embeddings = []
        for text in texts:
            # Use Gemini's embedding model
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        # Fallback to simple embeddings if Gemini fails
        return generate_simple_embeddings(texts)

def generate_simple_embeddings(texts: List[str]) -> List[List[float]]:
    """Fallback simple embeddings"""
    import hashlib
    embeddings = []
    for text in texts:
        # Create a simple embedding based on text characteristics
        text_lower = text.lower()
        features = [
            len(text) / 1000.0,
            text.count(' ') / 100.0,
            text.count('.') / 10.0,
            text.count('?') / 5.0,
            text.count('!') / 5.0,
        ]

        # Add hash-based features
        text_hash = hashlib.md5(text_lower.encode()).hexdigest()
        for i in range(0, len(text_hash), 2):
            hex_val = int(text_hash[i:i+2], 16)
            features.append((hex_val - 128) / 128.0)

        # Pad to 768 dimensions (Gemini embedding size)
        while len(features) < 768:
            features.append(0.0)
        features = features[:768]
        embeddings.append(features)

    return embeddings

def cosine_similarity_manual(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity without external dependencies"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)

# Document processing functions
async def download_document(url: str) -> bytes:
    """Download document from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        logger.error(f"Error downloading document from {url}: {e}")
        raise Exception(f"Failed to download document: {e}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content"""
    try:
        laparams = LAParams(
            boxes_flow=0.5,
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5
        )
        
        text = extract_text(
            io.BytesIO(pdf_content),
            laparams=laparams
        )
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise Exception(f"Failed to extract PDF text: {e}")

def extract_text_from_docx(docx_content: bytes) -> str:
    """Extract text from DOCX content"""
    try:
        doc = Document(io.BytesIO(docx_content))
        text_parts = []
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text_parts.append(cell.text)
        
        return '\n'.join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise Exception(f"Failed to extract DOCX text: {e}")

def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[dict]:
    """Split text into overlapping chunks with smart boundary detection"""
    chunks = []
    text_length = len(text)
    start = 0
    chunk_id = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to break at sentence boundary for better semantic coherence
        if end < text_length:
            # Look for sentence endings within the last 200 characters
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                sentence_end = text.rfind(punct, max(start, end - 200), end)
                if sentence_end > start:
                    end = sentence_end + len(punct)
                    break

        chunk_text = text[start:end].strip()

        if chunk_text and len(chunk_text) > 50:  # Only include meaningful chunks
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'start_pos': start,
                'end_pos': end,
                'length': len(chunk_text)
            })
            chunk_id += 1

        # Move start position with overlap
        start = end - overlap
        if start >= text_length:
            break

    return chunks

async def process_document(document_url: str) -> dict:
    """Process document from URL with embeddings"""
    try:
        # Download document
        logger.info(f"Downloading document from: {document_url}")
        content = await download_document(document_url)

        # Determine file type and extract text
        if str(document_url).lower().endswith('.pdf') or 'pdf' in str(document_url).lower():
            text = extract_text_from_pdf(content)
        elif str(document_url).lower().endswith('.docx'):
            text = extract_text_from_docx(content)
        else:
            # Try PDF first, then DOCX
            try:
                text = extract_text_from_pdf(content)
            except:
                text = extract_text_from_docx(content)

        # Clean and chunk text
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)

        # Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = generate_embeddings(chunk_texts)

        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]

        logger.info(f"Processed document: {len(chunks)} chunks, {len(cleaned_text)} characters")

        return {
            'url': document_url,
            'full_text': cleaned_text,
            'chunks': chunks,
            'total_chunks': len(chunks),
            'total_characters': len(cleaned_text),
            'embeddings': embeddings
        }

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise Exception(f"Document processing failed: {e}")

def find_relevant_chunks(query: str, document_data: dict, top_k: int = 5) -> List[dict]:
    """Find most relevant chunks using semantic similarity"""
    try:
        # Generate embedding for the query
        query_embedding = generate_embeddings([query])[0]

        # Calculate similarities with all chunks
        similarities = []
        for chunk in document_data['chunks']:
            if 'embedding' in chunk:
                similarity = cosine_similarity_manual(query_embedding, chunk['embedding'])
                similarities.append({
                    'chunk': chunk,
                    'similarity': similarity
                })

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        relevant_chunks = []
        for item in similarities[:top_k]:
            chunk_data = item['chunk'].copy()
            chunk_data['similarity_score'] = item['similarity']
            relevant_chunks.append(chunk_data)

        logger.info(f"Found {len(relevant_chunks)} relevant chunks with similarities: {[f'{c['similarity_score']:.3f}' for c in relevant_chunks]}")
        return relevant_chunks

    except Exception as e:
        logger.error(f"Error finding relevant chunks: {e}")
        return []

async def answer_question_with_context(question: str, relevant_chunks: List[dict]) -> str:
    """Generate answer using Gemini AI with relevant context chunks"""
    try:
        model = get_gemini_model()

        if model is None:
            return "Error: AI service is not available. Please try again later."

        if not relevant_chunks:
            return "This information is not available in the provided document."

        # Prepare context from relevant chunks
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            similarity = chunk.get('similarity_score', 0)
            context_parts.append(f"[Context {i+1} - Relevance: {similarity:.2f}]\n{chunk['text']}\n")

        context = "\n".join(context_parts)

        # Enhanced prompt for better accuracy
        prompt = f"""You are an expert document analyst specializing in insurance, legal, and policy documents. Your task is to answer questions based ONLY on the provided document context.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY based on the information provided in the document context above
2. If the specific information is not in the context, respond with "This information is not available in the provided document"
3. Include specific details, numbers, dates, percentages, and conditions when available
4. Quote relevant sections when appropriate
5. Be precise and factual - do not infer or assume information not explicitly stated
6. If multiple conditions or scenarios apply, list them clearly
7. For policy-related questions, include relevant clauses, waiting periods, coverage limits, etc.

ANSWER:"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.1
            )
        )

        if response and response.text:
            answer = response.text.strip()

            # Add source reference if answer contains information
            if "not available" not in answer.lower() and len(relevant_chunks) > 0:
                best_similarity = relevant_chunks[0].get('similarity_score', 0)
                if best_similarity > 0.3:  # Only add reference for good matches
                    answer += f"\n\n[Source: Document section with {best_similarity:.1%} relevance match]"

            return answer
        else:
            return "Error: No response generated from AI service"

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error: Unable to generate answer - {str(e)}"

# Document cache
document_cache = {}

async def process_queries(document_url: str, questions: List[str]) -> List[str]:
    """Process all queries for a document using semantic search"""
    try:
        # Check cache first
        if document_url in document_cache:
            logger.info("Using cached document data")
            document_data = document_cache[document_url]
        else:
            # Process document with embeddings
            logger.info("Processing new document with embeddings")
            document_data = await process_document(document_url)
            document_cache[document_url] = document_data

        # Answer all questions using semantic search
        answers = []
        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}: {question}")

                # Find relevant chunks using semantic similarity
                relevant_chunks = find_relevant_chunks(question, document_data, top_k=5)

                # Generate answer with context
                answer = await answer_question_with_context(question, relevant_chunks)
                answers.append(answer)

                logger.info(f"Generated answer for question {i+1}: {answer[:100]}...")

            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                answers.append(f"Error: Unable to process this question - {str(e)}")

        return answers

    except Exception as e:
        logger.error(f"Error in process_queries: {e}")
        return [f"Error: Document processing failed - {str(e)}"] * len(questions)

# Routes
@app.get("/", response_class=HTMLResponse)
async def frontend(request: Request):
    """Frontend interface for the LLM Query Retrieval System"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "LLM Query Retrieval System is running",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.get("/api/v1/hackrx/run")
async def hackrx_webhook_verification():
    """HackRX webhook endpoint - GET method for webhook verification"""
    return {
        "status": "success",
        "message": "HackRX webhook endpoint is active",
        "project_name": "LLM Query Retrieval System",
        "team": "AI Innovators",
        "hackrx_submission": True,
        "webhook_verified": True,
        "timestamp": time.time()
    }

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def hackrx_document_processing(
    request: QueryRequest,
    req: Request,
    token: str = Depends(verify_token)
):
    """Main endpoint for processing document queries"""
    start_time = time.time()

    try:
        logger.info(f"Processing request with {len(request.questions)} questions for document: {request.documents}")

        # Process the document and questions with timeout
        try:
            answers = await asyncio.wait_for(
                process_queries(
                    document_url=str(request.documents),
                    questions=request.questions
                ),
                timeout=25.0  # 25 second timeout
            )

            processing_time = time.time() - start_time
            logger.info(f"Request processed successfully in {processing_time:.2f} seconds")

        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.warning(f"Query processing timed out after {processing_time:.2f} seconds")
            timeout_message = "Processing timed out. The document may be too large or complex."
            answers = [timeout_message] * len(request.questions)

        return QueryResponse(answers=answers)

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing queries after {processing_time:.2f} seconds: {e}")
        error_message = f"Error processing queries: {str(e)}"
        error_answers = [error_message] * len(request.questions)
        return QueryResponse(answers=error_answers)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
