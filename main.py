"""
LLM-Powered Intelligent Query-Retrieval System
Main FastAPI application entry point
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
from dotenv import load_dotenv

# Import our modules with error handling for Vercel
try:
    from src.document_processor import DocumentProcessor
    from src.llm_service import LLMService
    from src.vector_service import VectorService
    from src.query_processor import QueryProcessor
    SERVICES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import services: {e}")
    SERVICES_AVAILABLE = False
    # Create dummy classes for basic functionality
    class DocumentProcessor:
        pass
    class LLMService:
        pass
    class VectorService:
        pass
    class QueryProcessor:
        def __init__(self, *args):
            pass
        async def process_queries(self, *args, **kwargs):
            return ["Error: Services not available in this environment"]

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

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str

# Global service instances (lazy-loaded)
document_processor = None
llm_service = None
vector_service = None
query_processor = None

def get_services():
    """Lazy-load services to avoid Vercel startup issues"""
    global document_processor, llm_service, vector_service, query_processor

    if query_processor is None:
        try:
            logger.info("Initializing services...")
            document_processor = DocumentProcessor()
            llm_service = LLMService()
            vector_service = VectorService()
            query_processor = QueryProcessor(document_processor, llm_service, vector_service)
            logger.info("Services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

    return query_processor

@app.get("/", response_class=HTMLResponse)
async def frontend(request: Request):
    """Frontend interface for the LLM Query Retrieval System"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Simple health check endpoint for Railway"""
    return {
        "status": "healthy",
        "message": "LLM Query Retrieval System is running",
        "timestamp": time.time(),
        "version": "1.0.0",
        "services_available": SERVICES_AVAILABLE
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0"
    )

@app.get("/api/v1/hackrx/run")
async def hackrx_webhook_status():
    """HackRX webhook endpoint - GET method for webhook verification"""
    return {
        "status": "success",
        "message": "HackRX webhook endpoint is active",
        "project_name": "LLM Query Retrieval System",
        "team": "AI Innovators",
        "hackrx_submission": True,
        "webhook_verified": True,
        "timestamp": time.time(),
        "endpoints": {
            "POST": "Document processing API",
            "GET": "Webhook verification"
        }
    }

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest,
    req: Request,
    token: str = Depends(verify_token)
):
    """
    Main endpoint for processing document queries

    Processes a document from a URL and answers natural language questions about it.
    Uses vector similarity search and LLM-based reasoning to provide accurate answers.
    """
    start_time = time.time()

    try:
        logger.info(f"Processing request with {len(request.questions)} questions for document: {request.documents}")

        # Validate request
        if len(request.questions) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one question is required"
            )

        # Get services (lazy-loaded)
        processor = get_services()

        # Process the document and questions with timeout
        try:
            answers = await asyncio.wait_for(
                processor.process_queries(
                    document_url=str(request.documents),
                    questions=request.questions
                ),
                timeout=25.0  # 25 second timeout to stay under Railway limits
            )

            processing_time = time.time() - start_time
            logger.info(f"Request processed successfully in {processing_time:.2f} seconds")

        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.warning(f"Query processing timed out after {processing_time:.2f} seconds")

            # Return timeout error for all questions
            timeout_message = "Processing timed out. The document may be too large or complex. Please try with a smaller document or fewer questions."
            answers = [timeout_message] * len(request.questions)

        # Ensure we have the same number of answers as questions
        if len(answers) != len(request.questions):
            logger.warning(f"Answer count mismatch: {len(answers)} answers for {len(request.questions)} questions")
            # Pad with error messages if needed
            while len(answers) < len(request.questions):
                answers.append("Error: Unable to generate answer for this question")

        return QueryResponse(answers=answers)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing queries after {processing_time:.2f} seconds: {e}")

        # Return structured error response
        error_message = f"Error processing queries: {str(e)}"
        error_answers = [error_message] * len(request.questions)

        return QueryResponse(answers=error_answers)

@app.post("/api/v1/preprocess")
async def preprocess_document(
    document_url: str,
    token: str = Depends(verify_token)
):
    """
    Preprocess a document for faster query processing
    """
    try:
        success = await query_processor.preprocess_document(document_url)
        return {"success": success, "message": "Document preprocessed successfully" if success else "Preprocessing failed"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error preprocessing document: {str(e)}"
        )

@app.delete("/api/v1/cache")
async def clear_cache(token: str = Depends(verify_token)):
    """Clear document cache"""
    try:
        query_processor.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing cache: {str(e)}"
        )

@app.get("/api/v1/hackrx/status")
async def hackrx_status():
    """HackRX project status endpoint for daily updates"""
    return {
        "status": "success",
        "project_name": "LLM Query Retrieval System",
        "team": "AI Innovators",
        "hackrx_submission": True,
        "progress": {
            "api_status": "deployed",
            "frontend_status": "deployed",
            "testing_status": "completed",
            "documentation_status": "completed"
        },
        "urls": {
            "api_endpoint": "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run",
            "demo_url": "https://llm-query-retrieval-system-production.up.railway.app",
            "documentation": "https://llm-query-retrieval-system-production.up.railway.app/docs"
        },
        "last_updated": time.time(),
        "features_completed": [
            "Document processing (PDF, DOCX)",
            "LLM-powered query parsing with Gemini AI",
            "Vector embeddings and semantic search",
            "Clause matching and logic evaluation",
            "Structured JSON responses",
            "Bearer token authentication",
            "Web frontend interface",
            "API documentation",
            "Sub-30 second response times",
            "HTTPS deployment on Railway"
        ],
        "technical_specs": {
            "backend": "FastAPI",
            "llm": "Google Gemini",
            "vector_db": "In-memory with fallback",
            "deployment": "Railway",
            "authentication": "Bearer token",
            "response_format": "JSON"
        }
    }

# Vercel handler
handler = app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
