"""
Query Processing Module
Main orchestrator for processing queries and generating answers
"""

import asyncio
import logging
from typing import List, Dict, Any
from .document_processor import DocumentProcessor
from .llm_service import LLMService
from .vector_service import VectorService

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Main query processing orchestrator"""
    
    def __init__(
        self, 
        document_processor: DocumentProcessor,
        llm_service: LLMService,
        vector_service: VectorService
    ):
        self.document_processor = document_processor
        self.llm_service = llm_service
        self.vector_service = vector_service
        
        # Cache for processed documents
        self.document_cache = {}
    
    async def process_queries(
        self, 
        document_url: str, 
        questions: List[str]
    ) -> List[str]:
        """Main method to process all queries for a document"""
        
        try:
            # Step 1: Process document (with caching)
            document_data = await self._get_or_process_document(document_url)
            
            # Step 2: Process all questions
            answers = []
            for question in questions:
                try:
                    answer = await self._process_single_query(
                        question, 
                        document_url, 
                        document_data
                    )
                    answers.append(answer)
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {e}")
                    answers.append(f"Error: Unable to process this question - {str(e)}")
            
            return answers
            
        except Exception as e:
            logger.error(f"Error in process_queries: {e}")
            # Return error for all questions
            return [f"Error: Document processing failed - {str(e)}"] * len(questions)
    
    async def _get_or_process_document(self, document_url: str) -> Dict[str, Any]:
        """Get document from cache or process it"""
        
        if document_url in self.document_cache:
            logger.info("Using cached document data")
            return self.document_cache[document_url]
        
        # Process document
        logger.info("Processing new document")
        document_data = await self.document_processor.process_document(document_url)
        
        # Generate embeddings for chunks
        chunk_texts = [chunk['text'] for chunk in document_data['chunks']]
        embeddings = await self.llm_service.generate_embeddings(chunk_texts)
        
        # Store embeddings in vector database
        await self.vector_service.store_document_embeddings(
            document_url,
            document_data['chunks'],
            embeddings
        )
        
        # Cache the processed document
        self.document_cache[document_url] = document_data
        
        return document_data
    
    async def _process_single_query(
        self, 
        question: str, 
        document_url: str, 
        document_data: Dict[str, Any]
    ) -> str:
        """Process a single query"""
        
        try:
            # Step 1: Parse query intent
            logger.info(f"Parsing query intent for: {question}")
            query_intent = await self.llm_service.parse_query_intent(question)
            
            # Step 2: Generate query embedding
            query_embeddings = await self.llm_service.generate_embeddings([question])
            query_embedding = query_embeddings[0]
            
            # Step 3: Find relevant chunks using vector similarity
            logger.info("Searching for relevant document chunks")
            relevant_chunks = await self.vector_service.search_similar_chunks(
                query_embedding,
                document_url,
                top_k=5
            )
            
            # Step 4: Enhance with keyword-based matching
            enhanced_chunks = await self._enhance_with_keyword_matching(
                question,
                query_intent,
                relevant_chunks,
                document_data['chunks']
            )
            
            # Step 5: Generate answer using LLM
            logger.info("Generating answer with LLM")
            answer = await self.llm_service.evaluate_answer_with_context(
                question,
                enhanced_chunks,
                query_intent
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing single query: {e}")
            return f"Error: Unable to process this question - {str(e)}"
    
    async def _enhance_with_keyword_matching(
        self,
        question: str,
        query_intent: Dict[str, Any],
        vector_chunks: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance vector search results with keyword-based matching"""
        
        try:
            # Get search terms from query intent
            search_terms = query_intent.get('search_terms', [])
            if not search_terms:
                return vector_chunks
            
            # Find additional chunks with keyword matches
            keyword_chunks = []
            vector_chunk_ids = {chunk['id'] for chunk in vector_chunks}
            
            for chunk in all_chunks:
                if chunk['id'] in vector_chunk_ids:
                    continue  # Already included from vector search
                
                chunk_text_lower = chunk['text'].lower()
                
                # Check for keyword matches
                matches = 0
                for term in search_terms:
                    if isinstance(term, str) and len(term) > 2:
                        if term.lower() in chunk_text_lower:
                            matches += 1
                
                # Include chunk if it has significant keyword matches
                if matches >= min(2, len(search_terms) // 2):
                    keyword_chunks.append({
                        **chunk,
                        'score': matches / len(search_terms)  # Simple scoring
                    })
            
            # Sort keyword chunks by score and take top ones
            keyword_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Combine vector and keyword results (limit total)
            combined_chunks = vector_chunks + keyword_chunks[:3]
            
            # Remove duplicates and limit total chunks
            seen_ids = set()
            final_chunks = []
            
            for chunk in combined_chunks:
                if chunk['id'] not in seen_ids:
                    seen_ids.add(chunk['id'])
                    final_chunks.append(chunk)
                    
                if len(final_chunks) >= 7:  # Limit total chunks
                    break
            
            logger.info(f"Enhanced search: {len(vector_chunks)} vector + {len(keyword_chunks)} keyword = {len(final_chunks)} total chunks")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error in keyword enhancement: {e}")
            return vector_chunks  # Fallback to vector results only
    
    def clear_cache(self):
        """Clear document cache"""
        self.document_cache.clear()
        logger.info("Document cache cleared")
    
    async def preprocess_document(self, document_url: str) -> bool:
        """Preprocess document for faster query processing"""
        try:
            await self._get_or_process_document(document_url)
            return True
        except Exception as e:
            logger.error(f"Error preprocessing document: {e}")
            return False
