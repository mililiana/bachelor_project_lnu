# pipline1/llm1.py
import json
import re
import os
from typing import List, Dict, Set, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from loguru import logger
import time
import google.generativeai as genai
from dotenv import load_dotenv
from hybrid_search import HybridSearchEngine
from utils import save_results_to_json
from prompt.build_prompt import build_system_prompt

load_dotenv()


class LLMQueryAnalyzer:

    def __init__(self, model="models/gemini-flash-latest"):
        self.model_name = model
        self.system_prompt = build_system_prompt()

        my_gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not my_gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=my_gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
        )

    def analyze(self, query: str) -> Dict:
        logger.info(
            f"Sending query to Gemini ('{self.model_name}') for analysis: '{query}'"
        )
        response = self.model.generate_content(query)

        raw_json = response.text
        logger.info(f"LLM (Gemini) analysis received: {raw_json}")

        parsed_json = json.loads(raw_json)

        if "filters" not in parsed_json:
            parsed_json["filters"] = None
        if "keywords" not in parsed_json or not isinstance(
            parsed_json["keywords"], list
        ):
            parsed_json["keywords"] = [query]

        return parsed_json


def main():
    DB_PATH = "vector_db"
    COLLECTION_NAME = "hybrid_collection"
    QUESTIONS_FILE_PATH = (
        "/Users/lilianamirchuk/Desktop/bachelor_project/evaluation/question.json"
    )
    RESULTS_OUTPUT_FILE = "evaluation/evaluation_results.json"
    all_run_results = []

    analyzer = LLMQueryAnalyzer()
    engine = HybridSearchEngine(db_path=DB_PATH, collection_name=COLLECTION_NAME)

    with open(QUESTIONS_FILE_PATH, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
        test_queries = [item["content"] for item in questions_data if "content" in item]

    for query in test_queries:
        print(f"QUERY: {query}")

        analysis = analyzer.analyze(query)

        print(
            f"-> LLM Plan: Filters={analysis['filters']}, Keywords={analysis['keywords']}"
        )
        print("-" * 80)

        results = engine.search(
            query_text=query,
            filters=analysis["filters"],
            keywords=analysis["keywords"],
            max_semantic_results=100,
        )

        run_data = {"query": query, "llm_plan": analysis, "search_results": results}

        all_run_results.append(run_data)

        if not results:
            print("  No results found.")
            print("\n" + "=" * 80 + "\n")
            continue

        for i, res in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Title:    {res['title']}")
            print(f"  Category: {res['category']}")
            print(f"  Content:  {res['content'][:100]}...")
            print(f"  Semantic: {res['semantic_score']:.4f}")
            print(f"  Boost:    {res['keyword_boost']:.4f}")
            print(f"  COMBINED: {res['combined_score']:.4f}  <--- (Sorted by this)")

        print("\n" + "=" * 80 + "\n")
        time.sleep(6)

    save_results_to_json(all_run_results, RESULTS_OUTPUT_FILE)


if __name__ == "__main__":
    main()
