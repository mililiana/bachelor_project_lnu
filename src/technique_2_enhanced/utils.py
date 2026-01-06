import json
from loguru import logger
from typing import List, Dict

def save_results_to_json(results_list: List[Dict], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=4)
        
        logger.info(f"{len(results_list)} результатів у файл: {filename}")
