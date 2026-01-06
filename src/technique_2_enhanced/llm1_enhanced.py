import json
import os
from typing import Dict
from loguru import logger
from groq import Groq
from dotenv import load_dotenv
from prompt.build_prompt_enhanced import build_enhanced_system_prompt

load_dotenv()


class EnhancedLLMQueryAnalyzer:
    """
    Enhanced query analyzer using Groq Llama 3.3 70B
    Determines query type and expected answer format using LLM reasoning
    """

    def __init__(self, model="meta-llama/llama-4-scout-17b-16e-instruct"):
        self.model_name = model
        self.system_prompt = build_enhanced_system_prompt()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.client = Groq(api_key=groq_api_key)

    def analyze(self, query: str) -> Dict:
        """
        Analyze query and extract:
        - filters (ChromaDB filters)
        - keywords (for boosting)
        - expected_answer_type (single/list/count)
        - explanation (reasoning)
        
        Returns:
            Dict with enhanced analysis including query type
        """
        logger.info(
            f"Sending query to Groq ('{self.model_name}') for enhanced analysis: '{query}'"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            raw_json = response.choices[0].message.content
            logger.info(f"LLM analysis received: {raw_json}")

            parsed_json = json.loads(raw_json)

            # Validate and set defaults
            if "filters" not in parsed_json:
                parsed_json["filters"] = None
            
            if "keywords" not in parsed_json or not isinstance(
                parsed_json["keywords"], list
            ):
                parsed_json["keywords"] = [query]
            
            # Ensure expected_answer_type exists
            if "expected_answer_type" not in parsed_json:
                parsed_json["expected_answer_type"] = "single"
            
            # Validate expected_answer_type values
            valid_types = ["single", "list", "count"]
            if parsed_json["expected_answer_type"] not in valid_types:
                logger.warning(
                    f"Invalid answer type '{parsed_json['expected_answer_type']}', "
                    f"defaulting to 'single'"
                )
                parsed_json["expected_answer_type"] = "single"
            
            logger.info(
                f"Query type determined by LLM: {parsed_json['expected_answer_type']}"
            )
            
            return parsed_json
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {raw_json}")
            # Return safe defaults
            return {
                "filters": None,
                "keywords": [query],
                "expected_answer_type": "single",
                "explanation": "Failed to parse LLM response"
            }
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return {
                "filters": None,
                "keywords": [query],
                "expected_answer_type": "single",
                "explanation": f"Error: {str(e)}"
            }


def main():
    """Test the enhanced analyzer"""
    analyzer = EnhancedLLMQueryAnalyzer()
    
    test_queries = [
        "Які факультети є в університеті?",  # list
        "Де знаходиться деканат ІКНІ?",  # single
        "Скільки кафедр на факультеті?",  # count
        "Перелічи всі гуртожитки",  # list
        "Як дістатися до головного корпусу?",  # single
        "Які спеціальності має кафедра програмування?",  # list
        "Хто декан факультету ІКНІ?",  # single
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = analyzer.analyze(query)
        
        print(f"Expected Answer Type: {result['expected_answer_type']}")
        print(f"Filters: {result.get('filters')}")
        print(f"Keywords: {result.get('keywords')}")
        if 'explanation' in result:
            print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    main()