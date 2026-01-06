import json
import os

BASE_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), 
    "system_prompt_enhanced_v3.txt"
)
METADATA_CACHE_PATH = (
    "/Users/lilianamirchuk/Desktop/bachelor_project/pipline1/prompt/"
    "vector_db_metadata_cache.json"
)


def extract_metadata_for_prompt():
    """Extract categories and titles from metadata cache"""
    if not os.path.exists(METADATA_CACHE_PATH):
        raise FileNotFoundError(
            f"Metadata cache not found: {METADATA_CACHE_PATH}. "
        )

    with open(METADATA_CACHE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    titles = data.get("titles", [])
    return sorted(categories), sorted(titles)


def build_enhanced_system_prompt():
    """Build system prompt with enhanced query type classification"""
    with open(BASE_PROMPT_PATH, "r", encoding="utf-8") as f:
        base_text = f.read()

    categories, titles = extract_metadata_for_prompt()

    final_prompt = base_text.format(
        categories="\n".join(f"- {c}" for c in categories),
        titles="\n".join(f"- {t}" for t in titles)
    )

    return final_prompt