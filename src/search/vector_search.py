import json
import re
from typing import List, Dict, Set
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
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        
        query_emb = self.model.encode([query])
        response = self.collection.query(
            query_embeddings=query_emb, 
            n_results=min(top_k * 3, 20)
        )
        
        results = []
        if not response or not response.get("ids", [[]])[0]:
            logger.warning("No semantic results found.")
            return []

        for i in range(len(response["ids"][0])):
            meta = response["metadatas"][0][i]
            semantic_score = 1 - response["distances"][0][i]
            
            combined_score = semantic_score
            
            results.append({
                "doc_id": meta.get("doc_id", ""),
                "title": meta.get("title", ""),
                "content": meta.get("content", ""),
                "category": meta.get("category", ""),
                "full_context": meta.get("full_context", ""),
                "semantic_score": semantic_score,
                "combined_score": combined_score
            })
        
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return results[:top_k]
