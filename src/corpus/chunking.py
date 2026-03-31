import json
import re
import os
from pathlib import Path
from typing import List, Dict
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class ImprovedChunking:
    
    @staticmethod
    def should_chunk(text: str, threshold: int = 800) -> bool:
        return len(text) > threshold
    
    @staticmethod
    def create_chunks_with_metadata(document: Dict) -> List[Dict]:
        title = document.get("title", "")
        content = document.get("content", "").strip()
        doc_id = document.get("doc_id", "unknown_doc")
        category = document.get("category", "")
        
        if not ImprovedChunking.should_chunk(content):
            return [{
                "doc_id": f"{doc_id}_chunk_001",
                "parent_doc_id": doc_id,
                "source_url": document.get("source_url", ""),
                "category": category,
                "title": title,
                "chunk_index": 1,
                "total_chunks": 1,
                "content": content,
                "full_context": f"{title}. {content}",

            }]
        
        chunks = ImprovedChunking.semantic_chunking(content)
        result = []
        
        for i, chunk_content in enumerate(chunks):
            result.append({
                "doc_id": f"{doc_id}_chunk_{i+1:03d}",
                "parent_doc_id": doc_id,
                "source_url": document.get("source_url", ""),
                "category": category,
                "title": title,
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
                "content": chunk_content,
                "full_context": f"{title}. {chunk_content}",

            })
        
        return result
    
    @staticmethod
    def semantic_chunking(text: str, chunk_size: int = 110,
                         chunk_overlap: int = 20) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence) + 1
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_len = len(s) + 1
                    if overlap_length + s_len <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_len
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

def main():
    input_file_path = str(PROJECT_ROOT / "data" / "meta_data_paraphrase_multilingual.json")
    output_file_path = str(PROJECT_ROOT / "data" / "chunked_documents_512.json")
    
    logger.info(f"Starting chunking process from {input_file_path}...")
    
    try:
        if not os.path.exists(input_file_path):
            logger.error(f"Input file not found: {input_file_path}")
            return
            
        with open(input_file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        if not isinstance(documents, list):
            logger.error(f"Input file format error: expected a JSON list of documents, got {type(documents)}")
            return

        logger.info(f"Loaded {len(documents)} documents.")
        
        all_chunks = []
        for doc in documents:
            try:
                chunks = ImprovedChunking.create_chunks_with_metadata(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to chunk document {doc.get('doc_id')}: {e}")
        
        logger.info(f"Created a total of {len(all_chunks)} chunks.")
        
        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            logger.warning(f"Output directory not found. Creating: {output_dir}")
            os.makedirs(output_dir)
            
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Successfully saved chunks to {output_file_path}")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file_path}")
    except PermissionError:
        logger.error(f"Permission denied. Could not write to {output_file_path}")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {input_file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()