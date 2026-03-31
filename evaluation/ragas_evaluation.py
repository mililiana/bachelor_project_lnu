"""
RAGAS evaluation for the RAG pipeline.

Evaluates retrieval and generation quality using RAGAS metrics:
  - Faithfulness: Is the answer grounded in the retrieved context?
  - Answer Relevancy: Does the answer address the question?
  - Context Precision: Are the retrieved documents relevant and well-ordered?
  - Context Recall: Did we retrieve all necessary information? (requires ground truth)

Uses OpenAI as the LLM judge (RAGAS default).
Ground truth answers from evaluation/questions/ground_truth.json enable Context Recall.

Thesis reference: §1.5 (RAGAS metrics description), §3.3-3.4 (evaluation results)

Usage:
    python -m evaluation.ragas_evaluation                              # evaluate latest results
    python -m evaluation.ragas_evaluation --input results/eval_semantic_key_all.json
    python -m evaluation.ragas_evaluation --run-pipeline               # run pipeline first, then evaluate
    python -m evaluation.ragas_evaluation --no-ground-truth            # skip context_recall
"""

import argparse
import json
import os
import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "questions" / "ground_truth.json"


def load_pipeline_results(results_path: str) -> list:
    """Load pre-computed pipeline results from JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth() -> dict:
    """
    Load ground truth answers from ground_truth.json.
    Returns a dict mapping original_question -> original_answer.
    """
    if not GROUND_TRUTH_PATH.exists():
        logger.warning(f"Ground truth file not found: {GROUND_TRUTH_PATH}")
        return {}

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_map = {}
    for item in data:
        question = item.get("original_question", "").strip()
        answer = item.get("original_answer", "").strip()
        if question and answer:
            gt_map[question] = answer

    logger.info(f"Loaded {len(gt_map)} ground truth answers")
    return gt_map


def run_pipeline_and_collect(questions_path: str) -> list:
    """
    Run the Gemini RAG pipeline on test questions and collect results.
    Returns list of dicts with query, search_results, generated_answer.
    """
    from src.generation.rag_pipeline_gemini import CompleteRAGSystem
    import time

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    rag = CompleteRAGSystem(
        db_path=str(PROJECT_ROOT / "vector_db"),
        collection_name="hybrid_collection",
    )

    results = []
    for i, q in enumerate(questions):
        query = q["content"]
        logger.info(f"[{i+1}/{len(questions)}] {query}")

        result = rag.query(query, return_sources=True)
        results.append({
            "query": query,
            "llm_plan": result.get("query_analysis", {}),
            "search_results": result.get("sources", []),
            "generated_answer": result.get("answer", ""),
            "num_retrieved": result.get("num_retrieved", 0),
        })
        time.sleep(6)  # rate limiting for Gemini

    return results


def prepare_ragas_dataset(pipeline_results: list, ground_truth: dict = None):
    """
    Convert pipeline results into RAGAS Dataset format.

    RAGAS expects:
      - question: str
      - answer: str
      - contexts: list[str]
      - ground_truth: str (optional, enables context_recall metric)
    """
    from datasets import Dataset

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    has_ground_truth = bool(ground_truth)

    for item in pipeline_results:
        query = item["query"]
        answer = item.get("generated_answer", "")
        ctx_list = []
        for doc in item.get("search_results", []):
            content = doc.get("content", "")
            title = doc.get("title", "")
            if content:
                ctx_list.append(f"{title}: {content}" if title else content)

        if not answer:
            answer = "No answer generated."
        if not ctx_list:
            ctx_list = ["No context retrieved."]

        questions.append(query)
        answers.append(answer)
        contexts.append(ctx_list)

        if has_ground_truth:
            gt = ground_truth.get(query, "")
            ground_truths.append(gt)

    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    if has_ground_truth:
        data_dict["ground_truth"] = ground_truths
        matched = sum(1 for gt in ground_truths if gt)
        logger.info(
            f"Ground truth matched for {matched}/{len(questions)} questions "
            f"({len(questions) - matched} without ground truth)"
        )

    dataset = Dataset.from_dict(data_dict)
    return dataset


def evaluate_with_ragas(dataset, output_path: str = None, use_ground_truth: bool = True):
    """
    Run RAGAS evaluation and return results.

    Requires OPENAI_API_KEY in environment.
    When ground_truth column is present and use_ground_truth=True,
    context_recall is included in the metrics.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error(
            "OPENAI_API_KEY not set. RAGAS uses OpenAI as the LLM judge.\n"
            "Set it in .env or export OPENAI_API_KEY=sk-..."
        )
        sys.exit(1)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]

    # Add context_recall only if ground truth is available
    has_gt = "ground_truth" in dataset.column_names
    if has_gt and use_ground_truth:
        metrics.append(context_recall)
        logger.info("Ground truth available — including context_recall metric")
    else:
        logger.info("No ground truth — evaluating without context_recall")

    logger.info(f"Running RAGAS evaluation on {len(dataset)} samples...")
    logger.info(f"Metrics: {[m.name for m in metrics]}")

    result = evaluate(dataset, metrics=metrics)

    logger.info("=== RAGAS Evaluation Results ===")
    for metric_name, score in result.items():
        if isinstance(score, (int, float)):
            logger.info(f"  {metric_name}: {score:.4f}")

    if output_path:
        output_data = {
            "metrics": {k: v for k, v in result.items() if isinstance(v, (int, float))},
            "num_samples": len(dataset),
            "ground_truth_used": has_gt and use_ground_truth,
        }

        # Include per-sample scores if available
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            output_data["per_sample"] = df.to_dict(orient="records")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for the RAG pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to pre-computed pipeline results JSON (relative to evaluation/)",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the Gemini pipeline on test questions before evaluating",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="questions/question_new_full.json",
        help="Path to questions file (relative to evaluation/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ragas_scores.json",
        help="Output path for RAGAS scores (relative to evaluation/)",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Skip context_recall even if ground truth is available",
    )
    args = parser.parse_args()

    eval_dir = Path(__file__).resolve().parent

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    # Load ground truth
    ground_truth = {} if args.no_ground_truth else load_ground_truth()

    if args.run_pipeline:
        logger.info("Running pipeline on test questions...")
        questions_path = str(eval_dir / args.questions)
        pipeline_results = run_pipeline_and_collect(questions_path)

        # Save pipeline results
        pipeline_output = str(eval_dir / "results" / "pipeline_results_latest.json")
        with open(pipeline_output, "w", encoding="utf-8") as f:
            json.dump(pipeline_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Pipeline results saved to {pipeline_output}")

    elif args.input:
        input_path = str(eval_dir / args.input)
        logger.info(f"Loading pipeline results from {input_path}")
        pipeline_results = load_pipeline_results(input_path)

    else:
        # Default: use the most complete existing results
        default_path = eval_dir / "results" / "eval_semantic_key_all.json"
        if not default_path.exists():
            default_path = eval_dir / "results" / "evaluation_results_with_answers.json"
        if not default_path.exists():
            logger.error(
                "No pipeline results found. Run with --run-pipeline or specify --input"
            )
            sys.exit(1)
        logger.info(f"Loading pipeline results from {default_path}")
        pipeline_results = load_pipeline_results(str(default_path))

    logger.info(f"Loaded {len(pipeline_results)} samples")

    dataset = prepare_ragas_dataset(pipeline_results, ground_truth)
    output_path = str(eval_dir / args.output)
    evaluate_with_ragas(dataset, output_path, use_ground_truth=not args.no_ground_truth)


if __name__ == "__main__":
    main()
