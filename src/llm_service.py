"""
LLM Service Module
Handles Google Gemini API integration for query parsing and logic evaluation
"""

import google.generativeai as genai
import os
import json
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with Google Gemini"""

    def __init__(self):
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.max_tokens = 4000
        self.temperature = 0.1  # Low temperature for consistent results

        # Initialize embedding model (using sentence-transformers as fallback)
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate for Gemini)"""
        # Approximate token count (Gemini uses different tokenization)
        return len(text.split()) * 1.3  # Rough approximation

    def truncate_context(self, context: str, max_tokens: int = 3000) -> str:
        """Truncate context to fit within token limits"""
        estimated_tokens = self.count_tokens(context)
        if estimated_tokens <= max_tokens:
            return context

        # Truncate by character count (approximate)
        char_ratio = max_tokens / estimated_tokens
        truncated_length = int(len(context) * char_ratio)
        return context[:truncated_length]
    
    async def parse_query_intent(self, question: str) -> Dict[str, Any]:
        """Parse user question to understand intent and extract key information"""
        
        system_prompt = """You are an expert at analyzing insurance and legal document queries. 
        Parse the user's question and extract:
        1. Main topic/subject (e.g., "grace period", "waiting period", "coverage")
        2. Specific details being asked (e.g., "premium payment", "pre-existing diseases")
        3. Question type (e.g., "definition", "condition", "coverage", "time_period", "amount")
        4. Key terms that should be searched for in the document
        
        Return a JSON object with these fields:
        - topic: main subject
        - details: specific aspects being asked
        - question_type: type of question
        - search_terms: list of important terms to search for
        - intent: brief description of what user wants to know
        """
        
        user_prompt = f"Question: {question}"
        
        try:
            response = await self._call_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=500
            )
            
            # Try to parse JSON response
            try:
                parsed_intent = json.loads(response)
                return parsed_intent
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "topic": "general",
                    "details": question,
                    "question_type": "general",
                    "search_terms": question.split(),
                    "intent": question
                }
                
        except Exception as e:
            logger.error(f"Error parsing query intent: {e}")
            # Return basic fallback
            return {
                "topic": "general",
                "details": question,
                "question_type": "general", 
                "search_terms": question.split(),
                "intent": question
            }
    
    async def evaluate_answer_with_context(
        self, 
        question: str, 
        relevant_chunks: List[Dict[str, Any]], 
        query_intent: Dict[str, Any]
    ) -> str:
        """Use LLM to evaluate and generate answer based on relevant document chunks"""
        
        # Prepare context from relevant chunks
        context_parts = []
        total_tokens = 0
        max_context_tokens = 2500
        
        for chunk in relevant_chunks:
            chunk_text = chunk.get('text', '')
            chunk_tokens = self.count_tokens(chunk_text)
            
            if total_tokens + chunk_tokens > max_context_tokens:
                break
                
            context_parts.append(f"[Chunk {chunk.get('id', 'N/A')}]: {chunk_text}")
            total_tokens += chunk_tokens
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are an expert insurance and legal document analyst. 
        Based on the provided document chunks, answer the user's question accurately and comprehensively.
        
        Guidelines:
        1. Only use information from the provided document chunks
        2. If the answer is not in the chunks, say "Information not found in the document"
        3. Be specific and cite relevant clauses or sections when possible
        4. For time periods, amounts, or conditions, be precise
        5. If there are multiple conditions or exceptions, list them clearly
        6. Keep the answer concise but complete
        
        Format your response as a clear, direct answer that addresses the specific question asked.
        """
        
        user_prompt = f"""Question: {question}

Query Intent: {json.dumps(query_intent, indent=2)}

Document Context:
{context}

Please provide a comprehensive answer based on the document context."""
        
        try:
            answer = await self._call_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=800
            )
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error: Unable to generate answer from the document."
    
    async def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """Make API call to Gemini"""

        try:
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\nUser Query: {user_prompt}"

            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature
            )

            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise Exception(f"LLM service error: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks using sentence-transformers"""

        try:
            if self.embedding_model is None:
                raise Exception("Embedding model not available")

            # Generate embeddings using sentence-transformers
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise Exception(f"Embedding generation failed: {e}")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text using sentence-transformers"""
        try:
            embeddings = await self.generate_embeddings([text])
            return embeddings[0]

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise Exception(f"Failed to generate embedding: {e}")
