import json
import re
from typing import List, Dict, Set, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from loguru import logger
import numpy as np

class HybridSearchEngine:

    
    def __init__(
        self,
        db_path: str = "vector_db",
        collection_name: str = "hybrid_collection",
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ):
        logger.info(f"Connecting to DB at {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    
    def search(
        self, 
        query_text: str, 
        filters: Optional[Dict] = None,  
        keywords: List[str] = None,     
        max_semantic_results: int = 100,
        top_k: int = None 
    ) -> List[Dict]:
        """
        Two-stage search approach:
        1. Semantic search first (retrieves up to max_semantic_results)
        2. Then applies keyword boosting to all retrieved results
        
        Args:
            query_text: User query
            filters: Optional ChromaDB filters (where, where_document)
            keywords: Keywords for boosting relevance (default: empty list)
            max_semantic_results: Maximum number of documents to retrieve in semantic search (default: 100)
            top_k: Deprecated parameter. If provided, will be used as max_semantic_results for backward compatibility
        
        Returns:
            List of all retrieved documents, sorted by combined_score (semantic + keyword boost)
        """
        # Backward compatibility: if top_k is provided, use it as max_semantic_results
        if top_k is not None:
            logger.warning(f"Parameter 'top_k' is deprecated. Use 'max_semantic_results' instead.")
            max_semantic_results = top_k
        
        # Default values
        if filters is None:
            filters = {}
        if keywords is None:
            keywords = []
        
        where_filter = filters.get("where") if filters else None
        where_document_filter = filters.get("where_document") if filters else None
        
        logger.info(f"Stage 1: Semantic search with max {max_semantic_results} results")
        logger.info(f"Applying Chroma metadata filters (where): {where_filter}")
        logger.info(f"Applying Chroma document filters (where_document): {where_document_filter}")
        query_emb = self.model.encode([query_text])

        # Get collection size to determine reasonable limit
        try:
            collection_count = self.collection.count()
            # Use min of requested max or collection size (with buffer)
            n_results = min(max_semantic_results, collection_count)
            logger.info(f"Collection has {collection_count} documents, retrieving up to {n_results}")
        except Exception as e:
            logger.warning(f"Could not get collection count: {e}, using {max_semantic_results}")
            n_results = max_semantic_results

        # Try progressive fallback strategy if filters are too strict
        response = None
        fallback_used = None
        
        # Strategy 1: Try with all filters (strictest)
        try:
            response = self.collection.query(
                query_embeddings=query_emb, 
                n_results=n_results, 
                where=where_filter,
                where_document=where_document_filter
            )
            if response and response.get("ids", [[]])[0]:
                logger.info(f"Semantic search successful with all filters: {len(response['ids'][0])} results")
        except Exception as e:
            logger.debug(f"Search with all filters failed: {e}")
        
        # Strategy 2: If no results, try without where_document filter (metadata only)
        if not response or not response.get("ids", [[]])[0]:
            if where_document_filter:
                logger.info("No results with where_document filter, trying without it...")
                try:
                    response = self.collection.query(
                        query_embeddings=query_emb, 
                        n_results=n_results, 
                        where=where_filter,
                        where_document=None
                    )
                    if response and response.get("ids", [[]])[0]:
                        fallback_used = "removed_where_document"
                        logger.info(f"Semantic search successful after removing where_document filter: {len(response['ids'][0])} results")
                except Exception as e:
                    logger.debug(f"Search without where_document filter failed: {e}")
        
        # Strategy 3: If still no results, try semantic search only (no filters)
        if not response or not response.get("ids", [[]])[0]:
            if where_filter or where_document_filter:
                logger.info("No results with filters, trying semantic search only...")
                try:
                    response = self.collection.query(
                        query_embeddings=query_emb, 
                        n_results=n_results, 
                        where=None,
                        where_document=None
                    )
                    if response and response.get("ids", [[]])[0]:
                        fallback_used = "semantic_only"
                        logger.info(f"Semantic search successful with no filters: {len(response['ids'][0])} results")
                except Exception as e:
                    logger.debug(f"Semantic-only search failed: {e}")
        
        results = []
        if not response or not response.get("ids", [[]])[0]:
            logger.warning("No semantic results found even after fallback strategies.")
            if fallback_used:
                logger.info(f"Fallback strategy used: {fallback_used}")
            return []

        logger.info(f"Stage 2: Applying keyword boosting to {len(response['ids'][0])} retrieved documents")
        logger.info(f"Keywords for boosting: {keywords}")
        keywords_lower = {k.lower() for k in keywords}

        # Apply keyword boosting to ALL retrieved documents
        for i in range(len(response["ids"][0])):
            meta = response["metadatas"][0][i]
            semantic_score = 1 - response["distances"][0][i]
            
            keyword_boost = 0.0
            title_lower = meta.get("title", "").lower()
            content_lower = meta.get("content", "").lower()

            # Calculate keyword boost
            for kw in keywords_lower:
                if kw in title_lower:
                    keyword_boost += 0.5
                elif kw in content_lower:
                    keyword_boost += 0.1 
            
            combined_score = semantic_score + keyword_boost
            
            results.append({
                "title": meta.get("title", ""),
                "content": meta.get("content", ""),
                "category": meta.get("category", ""),
                "semantic_score": semantic_score,
                "keyword_boost": keyword_boost,
                "combined_score": combined_score
            })
        
        # Sort by combined score (semantic + keyword boost)
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        logger.info(f"Returning {len(results)} documents sorted by combined_score (semantic + keyword boost)")
        # Return ALL results (no limit)
        return results
