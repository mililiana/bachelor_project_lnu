import json
import re
from typing import List, Dict, Set, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from loguru import logger
import numpy as np

class ImprovedHybridSearchEngine:
    """
    Enhanced hybrid search with LLM-based adaptive context selection
    """
    
    def __init__(
        self,
        db_path: str = "vector_db",
        collection_name: str = "hybrid_collection",
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ):
        logger.info(f"Connecting to DB at {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        # Use get_collection to connect to existing DB instead of creating new one
        try:
            self.collection = self.client.get_collection(name=collection_name)
            count = self.collection.count()
            logger.info(f"Connected to existing collection '{collection_name}' with {count} documents")
        except Exception as e:
            logger.warning(f"Collection not found, creating new one: {e}")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            count = self.collection.count()
            logger.info(f"Created new collection with {count} documents")
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def _calculate_keyword_boost(
        self, 
        text: str, 
        keywords: List[str],
        is_title: bool = False
    ) -> float:
        """
        Enhanced keyword boosting with:
        - Normalized scoring
        - Partial matching
        - Position awareness
        """
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        boost = 0.0
        
        for kw in keywords:
            kw_lower = kw.lower()
            
            # Use word boundary matching to avoid substring false positives
            # e.g., "кафе" should not match "кафедри"
            word_pattern = r'\b' + re.escape(kw_lower) + r'\b'
            
            # Exact match with word boundaries
            if re.search(word_pattern, text_lower):
                if is_title:
                    boost += 0.5  # Strong boost for title matches
                else:
                    # Check if keyword appears early in text (first 200 chars)
                    early_match = re.search(word_pattern, text_lower[:200]) is not None
                    boost += 1.5 if early_match else 1.0
            
            # Partial match (for compound multi-word keywords)
            elif len(kw_lower) > 4:  # Only for longer keywords
                kw_parts = kw_lower.split()
                if len(kw_parts) > 1:
                    # Each part should also match word boundaries
                    partial_matches = sum(1 for part in kw_parts 
                                        if re.search(r'\b' + re.escape(part) + r'\b', text_lower))
                    if partial_matches > 0:
                        boost += 0.2 * (partial_matches / len(kw_parts))
        
        # Normalize? No, we want strong boost if ANY important keyword is found.
        # But we still cap it.
        # normalized_boost = boost / max(len(keywords), 1) # This was diluting the score too much
        
        return min(boost, 3.0)  # Cap maximum boost (increased from 2.0)
    
    def _select_diverse_contexts(
        self, 
        results: List[Dict], 
        max_contexts: int,
        diversity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Select diverse contexts to avoid redundancy
        Uses category diversity and content similarity
        """
        if len(results) <= max_contexts:
            return results
        
        selected = [results[0]]  # Always include top result
        categories_used = {results[0]['category']}
        
        for doc in results[1:]:
            if len(selected) >= max_contexts:
                break
            
            # Prefer documents from different categories
            if doc['category'] not in categories_used:
                selected.append(doc)
                categories_used.add(doc['category'])
            # Or documents with high enough score
            elif doc['combined_score'] >= diversity_threshold:
                selected.append(doc)
        
        return selected
    
    def search(
        self, 
        query_text: str, 
        filters: Optional[Dict] = None,  
        keywords: List[str] = None,     
        max_semantic_results: int = 300,
        relevance_threshold: float = 0.3,
        max_context_docs: int = None,
        enable_diversity: bool = True,
        query_type_hint: str = None  # NEW: LLM's query type classification
    ) -> Tuple[List[Dict], Dict]:
        """
        Enhanced two-stage search with LLM-based adaptive context selection
        
        Args:
            query_text: User query
            filters: Optional ChromaDB filters
            keywords: Keywords for boosting
            max_semantic_results: Max docs in semantic search
            relevance_threshold: Min semantic score to keep doc
            max_context_docs: Max docs for LLM context (None = adaptive)
            enable_diversity: Use diversity-aware selection
            query_type_hint: Query type from LLM ('single', 'list', 'count')
        
        Returns:
            Tuple of (selected_contexts, metadata)
        """
        # Default values
        if filters is None:
            filters = {}
        if keywords is None:
            keywords = []
        
        # Use LLM's query type classification
        query_type = query_type_hint if query_type_hint else 'single'
        
        # Adaptive max context based on LLM's classification
        if max_context_docs is None:
            if query_type == 'list':
                max_context_docs = 15  # More docs for list queries
            elif query_type == 'count':
                max_context_docs = 10  # Moderate for count queries
            else:  # 'single'
                max_context_docs = 5   # Fewer docs for specific answers
        
        logger.info(f"Query type (LLM): {query_type}, Max contexts: {max_context_docs}")
        
        where_filter = filters.get("where") if filters else None
        where_document_filter = filters.get("where_document") if filters else None
        
        # Stage 1: Semantic search
        logger.info(f"Stage 1: Semantic search (max {max_semantic_results} results)")
        query_emb = self.model.encode([query_text])

        try:
            collection_count = self.collection.count()
            n_results = min(max_semantic_results, collection_count)
            logger.info(f"Collection: {collection_count} docs, retrieving: {n_results}")
        except Exception as e:
            logger.warning(f"Could not get collection count: {e}")
            n_results = max_semantic_results

        # Progressive fallback for filters
        response = None
        fallback_used = None
        
        # 1. Try with all filters
        try:
            response = self.collection.query(
                query_embeddings=query_emb, 
                n_results=n_results, 
                where=where_filter,
                where_document=where_document_filter
            )
            if response and response.get("ids", [[]])[0]:
                logger.info(f"Retrieved {len(response['ids'][0])} docs with all filters")
        except Exception as e:
            logger.debug(f"Search with all filters failed: {e}")
        
        # 2. Fallback: try without where_document
        if not response or not response.get("ids", [[]])[0]:
            if where_document_filter:
                logger.info("Retrying without where_document filter...")
                try:
                    response = self.collection.query(
                        query_embeddings=query_emb, 
                        n_results=n_results, 
                        where=where_filter,
                        where_document=None
                    )
                    if response and response.get("ids", [[]])[0]:
                        fallback_used = "removed_where_document"
                        logger.info(f"Retrieved {len(response['ids'][0])} docs without where_document")
                except Exception as e:
                    logger.debug(f"Fallback failed: {e}")
        
        # 3. Fallback: semantic only
        if not response or not response.get("ids", [[]])[0]:
            if where_filter or where_document_filter:
                logger.info("Retrying with semantic search only...")
                try:
                    response = self.collection.query(
                        query_embeddings=query_emb, 
                        n_results=n_results, 
                        where=None,
                        where_document=None
                    )
                    if response and response.get("ids", [[]])[0]:
                        fallback_used = "semantic_only"
                        logger.info(f"Retrieved {len(response['ids'][0])} docs (no filters)")
                except Exception as e:
                    logger.debug(f"Semantic-only search failed: {e}")

        # 4. KEYWORD EXPANSION (New)
        # Explicitly search for documents containing high-value keywords and merge them
        if keywords:
            logger.info(f"Performing KEYWORD EXPANSION for: {keywords}")
            try:
                # We need to ensure we have a valid response structure to merge into
                if not response or not response.get("ids", [[]])[0]:
                    response = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
                
                current_ids = set(response["ids"][0])
                
                for kw in keywords:
                    if len(kw) < 3: continue 
                    
                    # Search for documents containing the keyword
                    # Note: ChromaDB $contains doesn't support word boundaries, so we filter afterwards
                    kw_response = self.collection.query(
                        query_embeddings=query_emb, # Dummy, we rely on where_document
                        n_results=20, # Get more candidates for post-filtering
                        where_document={"$contains": kw}
                    )
                    
                    if kw_response and kw_response.get("ids", [[]])[0]:
                        new_ids = kw_response["ids"][0]
                        new_metas = kw_response["metadatas"][0]
                        new_dists = kw_response["distances"][0]
                        
                        # Post-filter with word boundaries to avoid false positives
                        word_pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                        
                        for i, doc_id in enumerate(new_ids):
                            if doc_id not in current_ids:
                                # Check if keyword matches with word boundaries
                                content = new_metas[i].get("content", "").lower()
                                if re.search(word_pattern, content):
                                    response["ids"][0].append(doc_id)
                                    response["metadatas"][0].append(new_metas[i])
                                    response["distances"][0].append(new_dists[i])
                                    current_ids.add(doc_id)
                                    logger.info(f"Added doc {doc_id} via keyword expansion ('{kw}')")
                                else:
                                    logger.debug(f"Filtered out doc {doc_id} - substring match only ('{kw}')")
                                
            except Exception as e:
                logger.error(f"Keyword expansion failed: {e}")

        if not response or not response.get("ids", [[]])[0]:
            logger.warning("No documents found")
            return [], {
                "query_type": query_type,
                "num_retrieved": 0,
                "num_filtered": 0,
                "fallback_used": fallback_used
            }

        # Stage 2: Enhanced keyword boosting and scoring
        logger.info(f"Stage 2: Enhanced keyword boosting and relevance filtering")
        logger.info(f"Keywords: {keywords}")
        
        all_results = []
        for i in range(len(response["ids"][0])):
            meta = response["metadatas"][0][i]
            semantic_score = 1 - response["distances"][0][i]
            
            # Skip documents below semantic threshold
            if semantic_score < relevance_threshold:
                continue
            
            # Calculate enhanced keyword boost
            title_boost = self._calculate_keyword_boost(
                meta.get("title", ""), 
                keywords, 
                is_title=True
            )
            content_boost = self._calculate_keyword_boost(
                meta.get("content", ""), 
                keywords, 
                is_title=False
            )
            
            keyword_boost = title_boost + content_boost
            
            # For list queries, boost keyword matches more
            boost_multiplier = 1.5 if query_type == 'list' else 1.0
            keyword_boost *= boost_multiplier
            
            combined_score = semantic_score + keyword_boost
            
            all_results.append({
                "title": meta.get("title", ""),
                "content": meta.get("content", ""),
                "category": meta.get("category", ""),
                "semantic_score": semantic_score,
                "keyword_boost": keyword_boost,
                "combined_score": combined_score
            })
        
        # Sort by combined score
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        logger.info(f"After relevance filtering: {len(all_results)} documents")
        
        # Stage 3: Adaptive context selection
        if len(all_results) == 0:
            return [], {
                "query_type": query_type,
                "num_retrieved": 0,
                "num_filtered": 0,
                "fallback_used": fallback_used
            }
        
        # For list queries with keyword matches, include all keyword-matched docs
        if query_type == 'list' and keywords:
            keyword_matched = [doc for doc in all_results if doc['keyword_boost'] > 0]
            if keyword_matched:
                # Take all keyword matches (up to reasonable limit)
                max_list_contexts = min(len(keyword_matched), 20)
                selected_contexts = keyword_matched[:max_list_contexts]
                logger.info(f"List query: selected {len(selected_contexts)} keyword-matched docs")
            else:
                # No keyword matches, use diverse top results
                selected_contexts = self._select_diverse_contexts(
                    all_results, 
                    max_context_docs,
                    diversity_threshold=0.5
                ) if enable_diversity else all_results[:max_context_docs]
        else:
            # Single answer query: use diverse top results
            selected_contexts = self._select_diverse_contexts(
                all_results, 
                max_context_docs,
                diversity_threshold=0.7
            ) if enable_diversity else all_results[:max_context_docs]
        
        logger.info(f"Final context selection: {len(selected_contexts)} documents")
        
        metadata = {
            "query_type": query_type,
            "num_retrieved": len(response["ids"][0]),
            "num_after_filtering": len(all_results),
            "num_selected_for_context": len(selected_contexts),
            "fallback_used": fallback_used,
            "all_results": all_results  # Keep for analysis
        }
        
        return selected_contexts, metadata