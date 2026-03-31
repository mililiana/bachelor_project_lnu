"""
Wilcoxon signed-rank test for comparing ranking methods.

Compares per-query RAGAS metrics between ranking methods to determine
if differences are statistically significant (p < 0.05).

Usage:
    python -m evaluation.statistical_tests \
        --baseline results/eval_semantic_key_all.json \
        --improved results/evaluation_results_pipeline2.json
"""

import argparse
import json
import sys
from pathlib import Path
from scipy import stats
import numpy as np
from loguru import logger


def load_per_sample_scores(path: str) -> dict:
    """
    Load per-sample RAGAS scores from evaluation output.
    Returns dict with metric_name -> list of scores.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "per_sample" in data:
        # Output from ragas_evaluation.py
        samples = data["per_sample"]
        metrics = {}
        for sample in samples:
            for key, val in sample.items():
                if isinstance(val, (int, float)) and key not in ("num_retrieved",):
                    metrics.setdefault(key, []).append(val)
        return metrics

    # Fallback: try to extract from raw pipeline results with inline scores
    logger.warning(f"No per_sample scores in {path}. Cannot run statistical test.")
    return {}


def wilcoxon_test(scores_a: list, scores_b: list, metric_name: str, alpha: float = 0.05):
    """
    Run Wilcoxon signed-rank test on paired samples.

    H0: No difference between methods.
    H1: There is a significant difference.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = b - a

    # Remove zero differences (ties)
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        logger.info(f"  {metric_name}: All differences are zero — methods are identical.")
        return

    stat, p_value = stats.wilcoxon(nonzero)

    mean_a = np.mean(a)
    mean_b = np.mean(b)
    significant = p_value < alpha

    logger.info(f"  {metric_name}:")
    logger.info(f"    Baseline mean:  {mean_a:.4f}")
    logger.info(f"    Improved mean:  {mean_b:.4f}")
    logger.info(f"    Wilcoxon stat:  {stat:.4f}")
    logger.info(f"    p-value:        {p_value:.6f}")
    logger.info(f"    Significant:    {'YES' if significant else 'NO'} (alpha={alpha})")

    return {
        "metric": metric_name,
        "baseline_mean": float(mean_a),
        "improved_mean": float(mean_b),
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
        "significant": significant,
    }


def main():
    parser = argparse.ArgumentParser(description="Wilcoxon signed-rank test for ranking methods")
    parser.add_argument("--baseline", required=True, help="Path to baseline RAGAS scores JSON")
    parser.add_argument("--improved", required=True, help="Path to improved RAGAS scores JSON")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    parser.add_argument("--output", type=str, default=None, help="Output path for test results")
    args = parser.parse_args()

    logger.info("Loading per-sample scores...")
    baseline = load_per_sample_scores(args.baseline)
    improved = load_per_sample_scores(args.improved)

    if not baseline or not improved:
        logger.error(
            "Cannot load per-sample scores. Run ragas_evaluation.py first "
            "to generate per-sample results."
        )
        sys.exit(1)

    # Find common metrics
    common_metrics = set(baseline.keys()) & set(improved.keys())
    if not common_metrics:
        logger.error("No common metrics found between the two files.")
        sys.exit(1)

    logger.info(f"Running Wilcoxon tests on: {sorted(common_metrics)}")
    logger.info("=" * 60)

    results = []
    for metric in sorted(common_metrics):
        a = baseline[metric]
        b = improved[metric]
        if len(a) != len(b):
            logger.warning(f"  {metric}: different sample counts ({len(a)} vs {len(b)}), skipping")
            continue
        result = wilcoxon_test(a, b, metric, args.alpha)
        if result:
            results.append(result)

    if args.output and results:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
