# University RAG System - Bachelor Thesis

Intelligent search and navigation system for university information resources based on Retrieval-Augmented Generation (RAG). Developed as part of a bachelor thesis at the National University "Lviv Polytechnic" (NULP).

## Overview

The system implements a hybrid search pipeline that combines semantic vector search with keyword-based retrieval to answer natural language questions about university information in Ukrainian. Three document ranking methods are compared: deterministic keyword boosting, XNet neural reranker, and KAN (Kolmogorov-Arnold Network) reranker.

## Project Structure

The codebase is organized to mirror the thesis chapters:

```
bachelor_project_lnu/
├── src/
│   ├── corpus/                        # Document corpus formation
│   │   ├── chunking.py                #   Semantic text chunking (110 tokens, 20 char overlap)
│   │   └── embeddings.py              #   Vectorization with paraphrase-multilingual-mpnet-base-v2
│   │
│   ├── search/                        # Search architecture & ranking
│   │   ├── vector_search.py           #   ChromaDB vector search interface
│   │   └── hybrid_search_basic.py     #   Hybrid search: semantic + keyword boosting
│   │
│   ├── query_analysis/                # Query understanding & intent detection
│   │   └── analyzer_gemini.py         #   Query analyzer using Google Gemini Flash
│   │
│   ├── generation/                    # Answer generation module
│   │   ├── rag_pipeline_gemini.py     #   Full RAG pipeline with Gemini (cloud)
│   │   └── lapa_llm.py               #   Local Lapa LLM v0.1.2 wrapper (12B, GPU)
│   │
│   ├── research/                      # Neural reranker models & metrics
│   │   ├── kan_model.py               #   KAN with B-spline learnable activations
│   │   ├── xnet_model.py              #   XNet with Cauchy activation function
│   │   └── ranking_metrics.py         #   NDCG@k, MRR, MAP, ERR@k metrics
│   │
│   ├── prompts/                       # System prompts for query analysis
│   │   ├── build_prompt_basic.py
│   │   └── system_prompt_base.txt
│   │
│   └── utils.py
│
├── web/                               # Web interface
│   ├── index.html                     #   Demo page for local testing
│   ├── widget.js                      #   Chat widget (injectable into any page)
│   └── widget.css                     #   Widget styles
│
├── app.py                             # Flask backend (connects widget to RAG pipeline)
│
├── evaluation/                        # Experiments & results
│   ├── ragas_evaluation.py            #   RAGAS metrics evaluation (OpenAI as judge)
│   ├── reranker_cv.py                 #   5-fold CV for KAN vs XNet vs baselines
│   ├── statistical_tests.py           #   Wilcoxon signed-rank test
│   ├── training_data_balanced.csv     #   Labeled training data for rerankers
│   ├── run_pipeline_groq.py           #   Alternative pipeline for evaluation experiments
│   ├── analyzer_groq.py               #   Groq-based query analyzer (evaluation only)
│   ├── hybrid_search_enhanced.py      #   Enhanced search variant (evaluation only)
│   ├── build_prompt_enhanced.py       #   Enhanced prompt builder (evaluation only)
│   ├── system_prompt_enhanced.txt     #   Enhanced system prompt (evaluation only)
│   ├── questions/                     #   110 validated test queries (5 categories)
│   └── results/                       #   RAGAS evaluation metrics & pipeline outputs
│
├── models/                            # Trained model checkpoints
│   ├── kan_model.pth                  #   KAN reranker weights
│   ├── xnet_model.pth                 #   XNet reranker weights
│   └── mlp_model.pth                  #   MLP baseline weights
│
├── config/
│   └── vector_db_metadata_cache.json  # Categories & titles for prompt building
│
├── data/
│   ├── processed_documents/           # 50+ cleaned university documents
│   ├── chunked_documents.json         # Document chunks
│   └── chunked_documents_128.json
│
├── vector_db/                         # ChromaDB persistent storage
├── plots/                             # Thesis figures
└── docs/                              # Additional documentation
```

## Thesis Chapter Mapping

| Directory | Thesis Section | Description |
|-----------|---------------|-------------|
| `src/corpus/` | Data collection, preprocessing, chunking, vectorization |
| `src/search/` | Search architecture, hybrid ranking methods |
| `src/query_analysis/` | Query understanding and intent detection |
| `src/generation/` | Answer generation (Gemini cloud + Lapa local) |
| `src/research/` | Neural rerankers: KAN (B-spline) & XNet (Cauchy) |
| `src/prompts/` | System prompts for LLM-based query analysis |
| `web/` + `app.py` | Web interface prototype (Flask + chat widget) |
| `evaluation/` | RAGAS evaluation, reranker CV, Wilcoxon tests |

## Technologies

| Component | Technology |
|-----------|-----------|
| Embedding model | `paraphrase-multilingual-mpnet-base-v2` |
| Vector database | ChromaDB (cosine similarity, HNSW index) |
| Cloud LLM | Google Gemini Flash (query analysis + answer generation) |
| Local LLM | Lapa LLM v0.1.2 (Gemma3-based, 12B params) |
| Web interface | Flask + vanilla JS/CSS widget (CORS-enabled) |
| Evaluation | RAGAS with OpenAI judge + Wilcoxon statistical test |
| GPU | NVIDIA RTX 4090 (24GB VRAM) via Vast.ai |

## Installation

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set:
- `GOOGLE_API_KEY` — for Gemini Flash (main pipeline)
- `OPENAI_API_KEY` — for RAGAS evaluation (LLM judge)
- `GROQ_API_KEY` — optional, for evaluation experiments only

## Usage

### Run the web interface (recommended for testing)

Start the Flask backend which serves the chat widget and connects it to the RAG pipeline:

```bash
# Using Gemini pipeline (cloud, default)
python app.py

# Using local Lapa LLM (requires GPU)
python app.py --pipeline lapa

# Custom port
python app.py --port 8080
```

Then open http://localhost:5001 in your browser. The chat widget appears in the bottom-right corner — type a question in Ukrainian and get answers from the university knowledge base.

### Run RAGAS evaluation

```bash
# Evaluate existing pipeline results
python -m evaluation.ragas_evaluation

# Evaluate a specific results file
python -m evaluation.ragas_evaluation --input results/eval_semantic_key_all.json

# Run the pipeline first, then evaluate
python -m evaluation.ragas_evaluation --run-pipeline
```

### Run reranker cross-validation (KAN vs XNet vs baselines)

```bash
python -m evaluation.reranker_cv
```

### Run Wilcoxon statistical test

```bash
# Compare two ranking methods
python -m evaluation.statistical_tests \
    --baseline evaluation/results/ragas_scores_baseline.json \
    --improved evaluation/results/ragas_scores_kan.json
```

### Embedding the widget into any webpage

The widget in `web/widget.js` is designed to be injected into any page (e.g. the university website). Just include the script tag:

```html
<script src="widget.js"></script>
```

It will automatically create the chat UI and send requests to `http://localhost:5001/ask`.

## Author

Liliana Mirchuk
Lviv Polytechnic National University, 2026
