"""
Vector Database Service Module
Handles Pinecone integration and vector similarity search
"""

import os
import hashlib
from typing import List, Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

class VectorService:
    """Service for vector database operations using Pinecone"""
    
    def __init__(self):
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.environment = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp-free')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'document-embeddings')
        self.dimension = 384  # Updated for sentence-transformers/fallback embeddings
        
        # Initialize Pinecone
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection and index"""
        # Always use in-memory vector search for reliability
        logger.info("Using in-memory vector search for maximum compatibility")
        self.use_pinecone = False
        self.vectors_cache = {}
    
    def _generate_chunk_id(self, document_url: str, chunk_id: int) -> str:
        """Generate unique ID for document chunk"""
        url_hash = hashlib.md5(document_url.encode()).hexdigest()[:8]
        return f"{url_hash}_chunk_{chunk_id}"

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity without numpy"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(a * a for a in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)
    
    async def store_document_embeddings(
        self, 
        document_url: str, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]]
    ):
        """Store document chunk embeddings in vector database"""
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        try:
            if self.use_pinecone:
                await self._store_in_pinecone(document_url, chunks, embeddings)
            else:
                await self._store_in_memory(document_url, chunks, embeddings)
                
            logger.info(f"Stored {len(chunks)} embeddings for document")
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise Exception(f"Failed to store embeddings: {e}")
    
    async def _store_in_pinecone(
        self, 
        document_url: str, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]]
    ):
        """Store embeddings in Pinecone"""
        
        vectors_to_upsert = []
        
        for chunk, embedding in zip(chunks, embeddings):
            vector_id = self._generate_chunk_id(document_url, chunk['id'])
            
            metadata = {
                'document_url': document_url,
                'chunk_id': chunk['id'],
                'text': chunk['text'][:1000],  # Limit metadata text size
                'start_pos': chunk['start_pos'],
                'end_pos': chunk['end_pos'],
                'length': chunk['length']
            }
            
            vectors_to_upsert.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    async def _store_in_memory(
        self, 
        document_url: str, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]]
    ):
        """Store embeddings in memory (fallback)"""
        
        for chunk, embedding in zip(chunks, embeddings):
            vector_id = self._generate_chunk_id(document_url, chunk['id'])
            
            self.vectors_cache[vector_id] = {
                'embedding': embedding,
                'metadata': {
                    'document_url': document_url,
                    'chunk_id': chunk['id'],
                    'text': chunk['text'],
                    'start_pos': chunk['start_pos'],
                    'end_pos': chunk['end_pos'],
                    'length': chunk['length']
                }
            }
    
    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        document_url: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        
        try:
            if self.use_pinecone:
                return await self._search_pinecone(query_embedding, document_url, top_k)
            else:
                return await self._search_memory(query_embedding, document_url, top_k)
                
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    async def _search_pinecone(
        self, 
        query_embedding: List[float], 
        document_url: str, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search using Pinecone"""
        
        # Create filter for specific document
        url_hash = hashlib.md5(document_url.encode()).hexdigest()[:8]
        
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={'document_url': document_url}
        )
        
        results = []
        for match in response['matches']:
            result = {
                'id': match['metadata']['chunk_id'],
                'text': match['metadata']['text'],
                'start_pos': match['metadata']['start_pos'],
                'end_pos': match['metadata']['end_pos'],
                'length': match['metadata']['length'],
                'score': match['score']
            }
            results.append(result)
        
        return results
    
    async def _search_memory(
        self, 
        query_embedding: List[float], 
        document_url: str, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search using in-memory vectors (fallback)"""
        
        similarities = []

        for vector_id, data in self.vectors_cache.items():
            if data['metadata']['document_url'] == document_url:
                stored_vector = data['embedding']

                # Calculate cosine similarity without numpy
                similarity = self._cosine_similarity(query_embedding, stored_vector)
                
                similarities.append({
                    'similarity': similarity,
                    'data': data['metadata']
                })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        results = []
        for item in similarities[:top_k]:
            metadata = item['data']
            result = {
                'id': metadata['chunk_id'],
                'text': metadata['text'],
                'start_pos': metadata['start_pos'],
                'end_pos': metadata['end_pos'],
                'length': metadata['length'],
                'score': item['similarity']
            }
            results.append(result)
        
        return results
    
    async def clear_document_embeddings(self, document_url: str):
        """Clear embeddings for a specific document"""
        
        try:
            if self.use_pinecone:
                # Delete vectors with matching document URL
                url_hash = hashlib.md5(document_url.encode()).hexdigest()[:8]
                self.index.delete(filter={'document_url': document_url})
            else:
                # Remove from memory cache
                keys_to_remove = [
                    k for k, v in self.vectors_cache.items() 
                    if v['metadata']['document_url'] == document_url
                ]
                for key in keys_to_remove:
                    del self.vectors_cache[key]
                    
            logger.info(f"Cleared embeddings for document: {document_url}")
            
        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")
            raise Exception(f"Failed to clear embeddings: {e}")
