# University RAG System - Bachelor Thesis

A Retrieval-Augmented Generation (RAG) system for university information retrieval, developed as part of a bachelor thesis comparing semantic search techniques.

## Overview

This project implements and compares two hybrid search configurations for a university information assistant:

- **Configuration A**: Similarity-based semantic retrieval followed by keyword filtering (narrowing down concepts)
- **Configuration B**: Keyword-based retrieval followed by semantic similarity filtering (contextualizing entities)

The system provides an intelligent Q&A interface for university-related information at the National University "Lviv Polytechnic" (NULP).

## Project Structure

```
├── src/                                # Source code
│   ├── data_processing/                # Data collection & processing scripts
│   │   ├── chunk.py                    # Document chunking with semantic splitting
│   │   └── create_and_save_embeddings.py
│   │
│   ├── technique_1_basic/              # Configuration A: Semantic First → Keyword Filtering
│   │   ├── llm1.py                     # Query analysis with Gemini
│   │   ├── hybrid_search.py            # Semantic search with keyword boosting
│   │   ├── vector_search_engine.py     # Vector DB interface
│   │   ├── complete_rag_system.py      # Full RAG pipeline
│   │   └── prompt/                     # System prompts
│   │
│   └── technique_2_enhanced/           # Configuration B: Keyword First → Semantic Filtering
│       ├── llm1_enhanced.py            # Enhanced query analysis with type classification
│       ├── improved_hybrid_search.py   # Keyword retrieval with semantic re-ranking
│       ├── improved_complete_rag_system.py
│       └── prompt/                     # Enhanced system prompts
│
├── data/                               # Processed data
│   ├── processed_documents/            # Cleaned text files (25 documents)
│   ├── chunked_documents.json          # Document chunks (128 tokens)
│   └── chunked_documents_512.json      # Document chunks (512 tokens)
│
├── vector_db/                          # ChromaDB vector database
│
├── evaluation/                         # Evaluation scripts and results
│   ├── questions/                      # Test question sets (30 queries)
│   └── results/                        # Evaluation metrics
│
├── plots/                              # Thesis figures and plots
│
└── docs/                               # Documentation
    └── Chapter_Research_Implementation.md
```

## Key Features

### Configuration A: Similarity First → Keyword Filtering
- **Stage 1**: Semantic search retrieves conceptually similar documents
- **Stage 2**: Keyword boosting refines and prioritizes results
- Query embedding using `paraphrase-multilingual-mpnet-base-v2`
- Optimal for queries requiring semantic understanding
- Example: "How do students connect to university Wi-Fi?"

### Configuration B: Keyword First → Semantic Filtering
- **Stage 1**: Keyword-based retrieval finds documents with exact terms
- **Stage 2**: Semantic similarity re-ranks for contextual relevance
- **Query Type Classification**: Single, List, or Count queries
- **Adaptive Context Selection**: 5-15 documents based on query type
- Optimal for specific entity queries (e.g., building numbers, institute names)
- Example: "Where is building 19?"

## Data Sources

The knowledge base contains 50+ documents including:
- Institutional information (institutes, departments)
- Regulatory documents (university codes, policies)
- Student services (scholarships, student cards, dining)
- Academic programs (Erasmus+, double degrees)
- Campus infrastructure (building addresses, facilities)

## Technologies

- **Embedding Model**: `paraphrase-multilingual-mpnet-base-v2` (multilingual support)
- **Vector Database**: ChromaDB with cosine similarity
- **LLM**: Google Gemini Flash (query analysis and answer generation)
- **Evaluation**: RAGAS metrics (faithfulness, answer relevancy, context precision/recall)

## Installation

```bash
pip install -r requirements.txt
```

Required API key:
- `GOOGLE_API_KEY` for Gemini Flash

## Evaluation

The system was evaluated on 30 test queries across 5 categories:
1. Navigation and Infrastructure
2. Educational process and academic rules
3. Scholarships
4. Student Services, Events and Organizations
5. Structure and Institutions

## License

This project was developed as part of a bachelor thesis at the National University "Lviv Polytechnic".

## Author

Liliana Mirchuk
