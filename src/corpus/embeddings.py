import json
from pathlib import Path
from loguru import logger
import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(input_file: str, collection_name: str, db_path: str):

    logger.info(f"Loading chunks from {input_file}...")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)

        if not isinstance(all_chunks, list) or not all_chunks:
            logger.error(f"File {input_file} is empty or not a JSON list.")
            return

        logger.info(f"Loaded {len(all_chunks)} chunks.")

        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)

        texts = [
            chunk.get("full_context", chunk.get("content", "")) for chunk in all_chunks
        ]
        logger.info("Generating embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True)

        ids = [chunk["doc_id"] for chunk in all_chunks]
        metadatas = [chunk.copy() for chunk in all_chunks]
        categories = sorted({chunk["category"] for chunk in all_chunks})
        titles = sorted({chunk["title"] for chunk in all_chunks})

        cache_data = {"categories": categories, "titles": titles}

        cache_path = str(PROJECT_ROOT / "config" / "vector_db_metadata_cache.json")
        with open(
            cache_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        logger.info("Saved categories + titles metadata cache successfully.")

        documents_content = [
            chunk.get("full_context", chunk.get("content", "")) for chunk in all_chunks
        ]

        logger.info(f"Connecting to ChromaDB at {db_path}...")
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        logger.info("Ingesting into ChromaDB...")
        try:
            collection.add(
                embeddings=embeddings.tolist(),
                documents=documents_content,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Successfully ingested {len(all_chunks)} chunks (add)")
        except Exception as e:
            logger.error(f"Error during ChromaDB ingestion (add): {e}")
            logger.info("Trying to upsert instead...")
            collection.upsert(
                embeddings=embeddings.tolist(),
                documents=documents_content,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Successfully ingested {len(all_chunks)} chunks (upsert)")

        logger.info("Embedding creation and ingestion complete.")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {input_file}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    INPUT_FILE = str(PROJECT_ROOT / "data" / "chunked_documents.json")
    COLLECTION_NAME = "hybrid_collection"
    DB_PATH = str(PROJECT_ROOT / "vector_db")

    logger.info("Starting embedding ingestion with hardcoded parameters...")
    main(input_file=INPUT_FILE, collection_name=COLLECTION_NAME, db_path=DB_PATH)
