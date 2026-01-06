import json
import os
from typing import Dict, List, Optional
from loguru import logger
from groq import Groq
import time
import random
from dotenv import load_dotenv

from improved_hybrid_search import ImprovedHybridSearchEngine
from llm1_enhanced import EnhancedLLMQueryAnalyzer
from utils import save_results_to_json

load_dotenv()


def call_groq_with_limit(func, *args, **kwargs):
    """Rate limiting wrapper for Groq"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            wait_time = random.uniform(5, 10)
            print(f"Rate limit hit. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            return func(*args, **kwargs)
        else:
            raise


class ImprovedRAGSystem:
    """
    Enhanced RAG system with Groq Llama 3.3 70B:
    - Adaptive context selection based on query type
    - Relevance filtering
    - Better keyword boosting
    - Diversity-aware selection
    """
    
    def __init__(
        self,
        db_path: str = "vector_db",
        collection_name: str = "hybrid_collection",
        query_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        answer_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        groq_api_key: str = None,
        temperature: float = 0.1
    ):
        # Load API key from environment if not provided
        if groq_api_key is None:
            groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        logger.info("Initializing Improved RAG System with Groq...")
        
        # Stage 1: Enhanced Query Analysis (with query type detection)
        self.query_analyzer = EnhancedLLMQueryAnalyzer(model=query_model)
        
        # Stage 2: Enhanced Document Retrieval
        self.search_engine = ImprovedHybridSearchEngine(
            db_path=db_path,
            collection_name=collection_name
        )
        
        # Stage 3: Answer Generation with Groq
        self.groq_client = Groq(api_key=groq_api_key)
        self.answer_model = answer_model
        self.temperature = temperature  # Store temperature for answer generation
        
        logger.info(f"✓ Improved RAG System initialized with Groq! (temperature={temperature})")
    
    def generate_answer(
        self,
        query: str,
        selected_contexts: List[Dict],
        metadata: Dict
    ) -> str:
        """
        Generate answer with enhanced context information using Groq
        """
        if not selected_contexts:
            return "Вибачте, не вдалося знайти відповідну інформацію в базі даних університету для вашого запитання. Спробуйте переформулювати запитання або зверніться до інформаційної служби університету."
        
        # Format contexts with relevance scores
        contexts = []
        for i, doc in enumerate(selected_contexts, 1):
            title = doc.get("title", "Без назви")
            content = doc.get("content", "")
            combined_score = doc.get("combined_score", 0)
            keyword_boost = doc.get("keyword_boost", 0)
            
            # Add score info for high-relevance docs
            score_info = f" [високорелевантний]" if keyword_boost > 0.5 else ""
            contexts.append(
                f"Документ {i} ({title}{score_info}):\n{content}"
            )
        
        context_text = "\n\n---\n\n".join(contexts)
        num_docs = len(selected_contexts)
        query_type = metadata.get("query_type", "single")
        
        # Adaptive prompt based on query type
        if query_type == 'list':
            instruction_addendum = """
- Якщо запитання вимагає списку (наприклад, "які факультети", "перелічи кафедри"), надай ПОВНИЙ перелік усіх згаданих елементів
- Організуй відповідь структуровано: використовуй нумерацію або маркери для списків
- Якщо є багато елементів, групуй їх за категоріями"""
        else:
            instruction_addendum = """
- Надай чітку, сфокусовану відповідь на конкретне запитання
- Уникай зайвої інформації, яка не стосується запитання безпосередньо"""
        
        prompt = f"""Ти - асистент для університетської інформаційної системи Національного університету "Львівська політехніка".
Твоє завдання - надати чітку, точну та лаконічну відповідь на запитання користувача українською мовою.

ВАЖЛИВО:
- Відповідай ТІЛЬКИ на основі наданих контекстів з бази знань університету
- Якщо інформація в контекстах недостатня для повної відповіді, скажи про це чесно і вкажи, що саме відсутнє
- Якщо жоден з документів не містить релевантної інформації, так і скажи
- Використовуй конкретні факти, числа, адреси, назви — З ТОЧНІСТЮ як вони зазначені в контекстах
- Будь лаконічним, але повним і інформативним{instruction_addendum}

Знайдено релевантних документів для контексту: {num_docs}
Тип запитання: {query_type}

Контексти з бази знань університету:
{context_text}

Запитання користувача: {query}

Відповідь (українською мовою):"""

        try:
            response = call_groq_with_limit(
                self.groq_client.chat.completions.create,
                model=self.answer_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,  # Use instance temperature
                max_tokens=2000
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Вибачте, сталася помилка при генерації відповіді."
    
    def query(
        self,
        user_query: str,
        return_sources: bool = False,
        max_semantic_results: int = 300,
        relevance_threshold: float = 0.3,
        max_context_docs: int = None,
        enable_diversity: bool = True
    ) -> Dict:
        """
        Complete enhanced RAG pipeline
        
        Args:
            user_query: User's question
            return_sources: Include source documents in response
            max_semantic_results: Max docs in semantic search (default: 100)
            relevance_threshold: Min semantic score to keep doc (default: 0.3)
            max_context_docs: Max docs for LLM context (None = adaptive)
            enable_diversity: Use diversity-aware selection
        
        Returns:
            Dict with answer, metadata, and optionally sources
        """
        logger.info(f"Processing query: {user_query}")
        
        # Stage 1: Analyze query (including query type)
        logger.info("Stage 1: Analyzing query...")
        query_analysis = self.query_analyzer.analyze(user_query)
        logger.info(f"Filters: {query_analysis.get('filters')}")
        logger.info(f"Keywords: {query_analysis.get('keywords')}")
        logger.info(f"Query type (LLM): {query_analysis.get('expected_answer_type')}")
        
        # Stage 2: Retrieve and select contexts
        logger.info("Stage 2: Enhanced retrieval with adaptive context selection...")
        
        # Use LLM's query type determination
        llm_query_type = query_analysis.get('expected_answer_type', 'single')
        
        selected_contexts, search_metadata = self.search_engine.search(
            query_text=user_query,
            filters=query_analysis.get("filters"),
            keywords=query_analysis.get("keywords"),
            max_semantic_results=max_semantic_results,
            relevance_threshold=relevance_threshold,
            max_context_docs=max_context_docs,
            enable_diversity=enable_diversity,
            query_type_hint=llm_query_type  # Pass LLM's classification
        )
        
        logger.info(f"Query type: {search_metadata.get('query_type')}")
        logger.info(f"Retrieved: {search_metadata.get('num_retrieved')} docs")
        logger.info(f"After filtering: {search_metadata.get('num_after_filtering')} docs")
        logger.info(f"Selected for context: {search_metadata.get('num_selected_for_context')} docs")
        
        if search_metadata.get('fallback_used'):
            logger.info(f"Fallback strategy: {search_metadata['fallback_used']}")
        
        # Stage 3: Generate answer
        logger.info("Stage 3: Generating answer with Groq...")
        answer = self.generate_answer(user_query, selected_contexts, search_metadata)
        logger.info("✓ Answer generated successfully")
        
        # Prepare response
        response = {
            "answer": answer,
            "query": user_query,
            "query_type": search_metadata.get("query_type"),
            "num_retrieved": search_metadata.get("num_retrieved"),
            "num_after_filtering": search_metadata.get("num_after_filtering"),
            "num_context_docs": len(selected_contexts),
            "fallback_used": search_metadata.get("fallback_used")
        }
        
        if return_sources:
            response["selected_contexts"] = selected_contexts
            response["all_results"] = search_metadata.get("all_results", [])
            response["query_analysis"] = query_analysis
        
        return response
    
    def query_simple(
        self, 
        user_query: str,
        max_semantic_results: int = 300,
        relevance_threshold: float = 0.1
    ) -> str:
        """Simplified interface returning only the answer"""
        result = self.query(
            user_query, 
            return_sources=False,
            max_semantic_results=max_semantic_results,
            relevance_threshold=relevance_threshold
        )
        return result["answer"]


def main():
    """
    Batch process questions with improved RAG system using Groq
    """
    # Configuration
    DB_PATH = "vector_db"
    COLLECTION_NAME = "hybrid_collection"
    QUESTIONS_FILE_PATH = (
        "/Users/lilianamirchuk/Desktop/bachelor_project/evaluation/question_new_full.json"
    )
    RESULTS_OUTPUT_FILE = (
        "/Users/lilianamirchuk/Desktop/bachelor_project/pipline_updated1_semantic_keywords_qroq/evaluation/"
        "evaluation_results_groq_llama4.json"
    )
    
    # Initialize improved RAG system
    logger.info("Initializing Improved RAG System with Groq...")
    rag = ImprovedRAGSystem(
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Load questions
    logger.info(f"Loading questions from {QUESTIONS_FILE_PATH}")
    try:
        with open(QUESTIONS_FILE_PATH, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        if isinstance(questions_data, list):
            test_queries = [item["content"] for item in questions_data if "content" in item]
        elif isinstance(questions_data, dict) and "questions" in questions_data:
            test_queries = questions_data["questions"]
        else:
            logger.error("Unknown JSON format")
            return
            
    except Exception as e:
        logger.error(f"Failed to load questions: {e}")
        return
    
    logger.info(f"Loaded {len(test_queries)} queries. Processing...")
    all_results = []
    
    # Process each query
    for idx, query in enumerate(test_queries, 1):
        print("\n" + "=" * 80)
        print(f"Processing Query {idx}/{len(test_queries)}: {query}")
        print("=" * 80)
        
        try:
            # Run improved RAG pipeline
            result = rag.query(
                query, 
                return_sources=True,
                relevance_threshold=0.1,  # Adjustable
                enable_diversity=True
            )
            
            # Prepare data structure
            run_data = {
                "query": query,
                "llm_plan": result.get("query_analysis", {}),
                "search_results": result.get("selected_contexts", []),
                "all_retrieved": result.get("all_results", []),
                "generated_answer": result.get("answer", ""),
                "query_type": result.get("query_type"),
                "num_retrieved": result.get("num_retrieved", 0),
                "num_filtered": result.get("num_after_filtering", 0),
                "num_context_docs": result.get("num_context_docs", 0),
                "fallback_used": result.get("fallback_used")
            }
            
            all_results.append(run_data)
            
            # Print detailed results
            print(f"\nRetrieval Stats:")
            print(f"  Query type: {result.get('query_type')}")
            print(f"  Retrieved: {result.get('num_retrieved')} docs")
            print(f"  After filtering: {result.get('num_after_filtering')} docs")
            print(f"  Used in context: {result.get('num_context_docs')} docs")
            if result.get('fallback_used'):
                print(f"  Fallback: {result['fallback_used']}")
            
            print(f"\nGenerated Answer:\n{result['answer']}")
            
            if result.get("selected_contexts"):
                print(f"\nContext Documents ({len(result['selected_contexts'])}):")
                for i, doc in enumerate(result['selected_contexts'], 1):
                    print(f"\n  {i}. {doc.get('title', 'No title')}")
                    print(f"     Category: {doc.get('category', 'N/A')}")
                    print(f"     Combined Score: {doc.get('combined_score', 0):.4f}")
                    print(f"       ├─ Semantic: {doc.get('semantic_score', 0):.4f}")
                    print(f"       └─ Keyword: +{doc.get('keyword_boost', 0):.4f}")
                    print(f"     Preview: {doc.get('content', '')[:120]}...")
            
            # Rate limiting (Groq is fast, but be respectful)
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            all_results.append({
                "query": query,
                "llm_plan": {},
                "search_results": [],
                "generated_answer": f"Error: {str(e)}",
                "error": str(e)
            })
            continue
    
    # Save results
    logger.info(f"Saving results to {RESULTS_OUTPUT_FILE}")
    save_results_to_json(all_results, RESULTS_OUTPUT_FILE)
    
    print("\n" + "=" * 80)
    print(f"COMPLETE! Processed {len(all_results)} queries.")
    print(f"Results saved to: {RESULTS_OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()