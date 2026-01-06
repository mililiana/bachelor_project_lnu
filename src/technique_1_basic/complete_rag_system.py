# pipline1/complete_rag_system.py
import json
import os
from typing import Dict, List, Optional
from loguru import logger
import google.generativeai as genai
import time
import random
from dotenv import load_dotenv
from hybrid_search_usekeywords import HybridSearchEngine
from llm1 import LLMQueryAnalyzer
from utils import save_results_to_json

load_dotenv()

def call_gemini_with_limit(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if "429" in str(e):
            wait_time = random.uniform(35, 45)
            print(f" Rate limit hit. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            return func(*args, **kwargs)
        else:
            raise


class CompleteRAGSystem:
    """
    Complete RAG system that:
    1. Analyzes user queries (using Gemini)
    2. Retrieves relevant documents (using hybrid search)
    3. Generates natural language answers (using Gemini)
    """
    
    def __init__(
        self,
        db_path: str = "vector_db",
        collection_name: str = "hybrid_collection",
        query_model: str = "models/gemini-flash-latest",
        answer_model: str = "models/gemini-flash-latest",
        gemini_api_key: str = None
    ):
        # Load API key from environment if not provided
        if gemini_api_key is None:
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        """
        Initialize the complete RAG system
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection
            query_model: Gemini model for query analysis
            answer_model: Gemini model for answer generation
            gemini_api_key: Google Gemini API key
        """
        logger.info("Initializing Complete RAG System...")
        
        # Stage 1: Query Analysis
        self.query_analyzer = LLMQueryAnalyzer(model=query_model)
        
        # Stage 2: Document Retrieval
        self.search_engine = HybridSearchEngine(
            db_path=db_path,
            collection_name=collection_name
        )
        
        # Stage 3: Answer Generation
        genai.configure(api_key=gemini_api_key)
        self.answer_model = genai.GenerativeModel(
            model_name=answer_model,
            generation_config={
                "temperature": 0.1,  # Lower temperature for more consistent answers
            },
        )
        
        logger.info("RAG System initialized successfully!")
    
    def generate_answer(
        self,
        query: str,
        retrieved_documents: List[Dict]
    ) -> str:
        """
        Generate a natural language answer from retrieved documents
        
        Args:
            query: User's question
            retrieved_documents: List of retrieved document dicts with 'title' and 'content'
                               Uses ALL retrieved documents dynamically
        
        Returns:
            Generated answer as a string
        """
        if not retrieved_documents:
            return "Вибачте, не вдалося знайти відповідну інформацію в базі даних університету для вашого запитання. Спробуйте переформулювати запитання або зверніться до інформаційної служби університету."
        
        # Format contexts (use ALL retrieved documents, no limit)
        contexts = []
        for i, doc in enumerate(retrieved_documents, 1):
            title = doc.get("title", "Без назви")
            content = doc.get("content", "")
            score = doc.get("combined_score", 0)
            # Include relevance score for context
            contexts.append(f"Документ {i} ({title}, релевантність: {score:.3f}):\n{content}")
        
        context_text = "\n\n---\n\n".join(contexts)
        num_docs = len(retrieved_documents)
        
        # Create prompt for answer generation
        prompt = f"""Ти - асистент для університетської інформаційної системи Національного університету "Львівська політехніка".
Твоє завдання - надати чітку, точну та лаконічну відповідь на запитання користувача українською мовою.

ВАЖЛИВО:
- Відповідай ТІЛЬКИ на основі наданих контекстів з бази знань університету
- Якщо інформації в контекстах недостатньо для повної відповіді, скажи про це чесно і вкажи, що саме відсутнє
- Якщо жоден з документів не містить релевантної інформації, так і скажи
- Використовуй конкретні факти, числа, адреси, назви З ТОЧНІСТЮ як вони зазначені в контекстах
- Будь лаконічним, але повним і інформативним
- Якщо в контекстах є кілька варіантів або додаткова інформація - згадай про це

Знайдено релевантних документів: {num_docs}

Контексти з бази знань університету:
{context_text}

Запитання користувача: {query}

Відповідь (українською мовою):"""

        try:
            response = call_gemini_with_limit(self.answer_model.generate_content, prompt)

            answer = response.text.strip()
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Вибачте, сталася помилка при генерації відповіді."
    
    def query(
        self,
        user_query: str,
        return_sources: bool = False,
        max_semantic_results: int = 100
    ) -> Dict:
        """
        Complete RAG pipeline: analyze query, retrieve documents, generate answer
        
        Uses two-stage approach:
        1. Semantic search first (retrieves up to max_semantic_results)
        2. Keyword boosting applied to all retrieved results
        
        Args:
            user_query: User's question
            return_sources: If True, include source documents in response
            max_semantic_results: Maximum number of documents to retrieve in semantic search (default: 100)
        
        Returns:
            Dict with:
            - 'answer': Generated answer
            - 'sources': (optional) List of ALL source documents (no limit)
            - 'query_analysis': (optional) Query analysis details
            - 'num_retrieved': Number of documents actually retrieved
        """
        logger.info(f"Processing query: {user_query}")
        
        # Stage 1: Analyze query
        logger.info("Stage 1: Analyzing query...")
        query_analysis = self.query_analyzer.analyze(user_query)
        logger.info(f"Generated filters: {query_analysis.get('filters')}")
        logger.info(f"Extracted keywords: {query_analysis.get('keywords')}")
        
        # Stage 2: Retrieve documents using two-stage approach
        # Stage 2a: Semantic search (first)
        # Stage 2b: Keyword boosting (applied to all semantic results)
        logger.info(f"Stage 2: Retrieving documents (semantic search + keyword boosting)...")
        logger.info(f"Max semantic results: {max_semantic_results}")
        retrieved_docs = self.search_engine.search(
            query_text=user_query,
            filters=query_analysis.get("filters"),
            keywords=query_analysis.get("keywords"),
            max_semantic_results=max_semantic_results
        )
        num_retrieved = len(retrieved_docs)
        
        if num_retrieved == 0:
            logger.warning("No documents retrieved - filters may be too strict or information not available")
        else:
            logger.info(f"Retrieved {num_retrieved} documents (all documents, sorted by combined_score)")
        
        # Stage 3: Generate answer (uses ALL retrieved documents)
        logger.info(f"Stage 3: Generating answer from {num_retrieved} document(s)...")
        answer = self.generate_answer(user_query, retrieved_docs)
        logger.info("Answer generated successfully")
        
        # Prepare response
        response = {
            "answer": answer,
            "query": user_query,
            "num_retrieved": num_retrieved
        }
        
        if return_sources:
            response["sources"] = retrieved_docs
            response["query_analysis"] = query_analysis
        
        return response
    
    def query_simple(self, user_query: str, max_semantic_results: int = 100) -> str:
        """
        Simplified interface that returns only the answer string
        
        Args:
            user_query: User's question
            max_semantic_results: Maximum number of documents to retrieve in semantic search (default: 100)
        
        Returns:
            Generated answer as a string
        """
        result = self.query(user_query, return_sources=False, max_semantic_results=max_semantic_results)
        return result["answer"]


def main():
    """
    Batch process questions from JSON file (similar to llm1.py)
    Processes queries through complete RAG pipeline and saves results with answers
    """
    # Configuration (same as llm1.py)
    DB_PATH = "vector_db"
    COLLECTION_NAME = "hybrid_collection"
    QUESTIONS_FILE_PATH = (
        "/Users/lilianamirchuk/Desktop/bachelor_project/evaluation/question_full.json"
    )
    RESULTS_OUTPUT_FILE = "/Users/lilianamirchuk/Desktop/bachelor_project/evaluation/evaluation_results_with_answers_keywords.json"
    
    # Initialize RAG system
    logger.info("Initializing Complete RAG System...")
    rag = CompleteRAGSystem(
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Load questions from JSON file
    logger.info(f"Loading questions from {QUESTIONS_FILE_PATH}")
    try:
        with open(QUESTIONS_FILE_PATH, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        # Extract queries (handle different JSON formats)
        if isinstance(questions_data, list):
            test_queries = [item["content"] for item in questions_data if "content" in item]
        elif isinstance(questions_data, dict) and "questions" in questions_data:
            test_queries = questions_data["questions"]
        else:
            logger.error("Unknown JSON format in question file")
            return
            
    except FileNotFoundError:
        logger.error(f"Question file not found: {QUESTIONS_FILE_PATH}")
        return
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {QUESTIONS_FILE_PATH}")
        return
    except Exception as e:
        logger.error(f"Failed to read question file: {e}")
        return
    
    logger.info(f"Loaded {len(test_queries)} queries. Processing...")
    all_results = []
    
    # Process each query
    for idx, query in enumerate(test_queries, 1):
        print("\n" + "=" * 80)
        print(f"Processing Query {idx}/{len(test_queries)}: {query}")
        print("=" * 80)
        
        try:
            # Run complete RAG pipeline
            result = rag.query(query,  return_sources=True)
            
            # Prepare data structure (compatible with ragas_evaluation.py format)
            run_data = {
                "query": query,
                "llm_plan": result.get("query_analysis", {}),
                "search_results": result.get("sources", []),
                "generated_answer": result.get("answer", ""),
                "num_retrieved": result.get("num_retrieved", 0)
            }
            
            all_results.append(run_data)
            
            # Print results
            num_retrieved = result.get("num_retrieved", 0)
            print(f"\nRetrieved: {num_retrieved} document(s)")
            print(f"\nGenerated Answer:\n{result['answer']}")
            
            if result.get("sources"):
                print(f"\nSource Documents ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n  {i}. {source.get('title', 'No title')}")
                    print(f"     Category: {source.get('category', 'N/A')}")
                    print(f"     Relevance Score: {source.get('combined_score', 0):.4f}")
                    print(f"       - Semantic: {source.get('semantic_score', 0):.4f}")
                    print(f"       - Keyword Boost: {source.get('keyword_boost', 0):.4f}")
                    print(f"     Content preview: {source.get('content', '')[:150]}...")
            else:
                print("\n    No documents retrieved - answer generated based on no context.")
            
            # Rate limiting (optional, adjust as needed)
            time.sleep(10)
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            # Still save partial results
            all_results.append({
                "query": query,
                "llm_plan": {},
                "search_results": [],
                "generated_answer": f"Error: {str(e)}"
            })
            continue
    
    # Save all results
    logger.info(f"Saving results to {RESULTS_OUTPUT_FILE}")
    save_results_to_json(all_results, RESULTS_OUTPUT_FILE)
    
    print("\n" + "=" * 80)
    print(f"COMPLETE! Processed {len(all_results)} queries.")
    print(f"Results saved to: {RESULTS_OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()

