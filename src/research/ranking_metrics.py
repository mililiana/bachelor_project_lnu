"""
Ranking Quality Metrics for Reranking Evaluation.

Implements the "gold standard" metrics for evaluating document ranking:
- NDCG@k (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)

All functions accept relevance scores in predicted rank order, supporting
both binary (0/1) and graded relevance (0.0 to 1.0).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at rank k.

    DCG@k = Σ_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)

    Args:
        relevance_scores: Relevance scores in predicted rank order
        k: Cutoff rank

    Returns:
        DCG value (higher is better)
    """
    scores = np.array(relevance_scores[:k], dtype=np.float64)
    if len(scores) == 0:
        return 0.0

    # Gains: 2^rel - 1 (works for both binary and graded)
    gains = np.power(2.0, scores) - 1.0

    # Discounts: log2(rank + 1), where rank is 1-indexed
    discounts = np.log2(np.arange(1, len(scores) + 1) + 1)

    return float(np.sum(gains / discounts))


def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank k.

    NDCG@k = DCG@k / IDCG@k

    The industry standard for ranking evaluation. Uses logarithmic
    discount to heavily reward relevant documents at the very top.

    Args:
        relevance_scores: Relevance scores in predicted rank order
        k: Cutoff rank

    Returns:
        NDCG@k in [0, 1] (1.0 = perfect ranking)
    """
    # Actual DCG
    actual_dcg = dcg_at_k(relevance_scores, k)

    # Ideal DCG: sort by relevance descending
    ideal_order = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_order, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def reciprocal_rank(relevance_scores: List[float], threshold: float = 0.5) -> float:
    """
    Compute Reciprocal Rank for a single query.

    RR = 1 / rank_of_first_relevant_document

    Focuses entirely on finding the first relevant result.
    If the first relevant doc is at rank n, score is 1/n.

    Args:
        relevance_scores: Relevance scores in predicted rank order
        threshold: Minimum score to consider a document "relevant"

    Returns:
        Reciprocal rank in (0, 1] or 0.0 if no relevant doc found
    """
    for i, score in enumerate(relevance_scores):
        if score >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(relevance_scores: List[float], threshold: float = 0.5) -> float:
    """
    Compute Average Precision for a single query.

    AP = (1/R) * Σ_{k=1}^{n} P(k) * rel(k)

    Where R = total relevant docs, P(k) = precision at rank k,
    rel(k) = 1 if doc at rank k is relevant.

    Best when the LLM needs to synthesize information from
    multiple relevant documents across the ranked list.

    Args:
        relevance_scores: Relevance scores in predicted rank order
        threshold: Minimum score to consider a document "relevant"

    Returns:
        AP in [0, 1] (1.0 = all relevant docs at the top)
    """
    relevant = [1 if s >= threshold else 0 for s in relevance_scores]
    total_relevant = sum(relevant)

    if total_relevant == 0:
        return 0.0

    cumulative_precision = 0.0
    relevant_count = 0

    for i, is_relevant in enumerate(relevant):
        if is_relevant:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            cumulative_precision += precision_at_i

    return cumulative_precision / total_relevant


def compute_ranking_metrics(
    relevance_scores: List[float],
    ks: List[int] = [1, 3, 5, 10],
    relevance_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all ranking metrics for a single query.

    Args:
        relevance_scores: Relevance scores in predicted rank order
        ks: List of cutoff values for NDCG@k
        relevance_threshold: Threshold for binary relevance in MRR/MAP

    Returns:
        Dict with keys like 'ndcg@1', 'ndcg@3', ..., 'mrr', 'map'
    """
    metrics = {}

    # NDCG at various k
    for k in ks:
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevance_scores, k)

    # ERR (Expected Reciprocal Rank)
    # For binary data (0/1), standard ERR (max_grade=1) caps satisfaction at 0.5.
    # To make it comparable to MRR (where 1=Perfect), we map 1 -> 4 (max_grade=4).
    # This yields P(satisfaction) = (2^4 - 1)/2^4 ~ 0.94

    # Scale scores: if binary max is 1, scale to 4
    max_rel = max(relevance_scores) if relevance_scores else 0
    scaled_scores = relevance_scores
    max_g = max_rel

    if max_rel <= 1.0:
        scaled_scores = [s * 4 for s in relevance_scores]
        max_g = 4.0

    for k in ks:
        metrics[f"err@{k}"] = err_at_k(scaled_scores, k=k, max_grade=max_g)

    # MRR
    metrics["mrr"] = reciprocal_rank(relevance_scores, relevance_threshold)

    # MAP (Average Precision for this query)
    metrics["map"] = average_precision(relevance_scores, relevance_threshold)

    return metrics


def err_at_k(relevance_scores: List[float], k: int, max_grade: float = 1.0) -> float:
    """
    Compute Expected Reciprocal Rank (ERR) at rank k.

    ERR = Sum_{r=1..k} (1/r) * P(user stops at r) * Product_{i=1..r-1} (1 - P(user stops at i))

    Where P(user stops at r) = R_r = (2^{rel_r} - 1) / 2^{max_grade}

    Args:
        relevance_scores: Relevance scores in predicted rank order
        k: Cutoff rank
        max_grade: Maximum possible relevance score (default 1.0 for binary 0/1)

    Returns:
        ERR value
    """
    scores = relevance_scores[:k]
    p = 1.0
    err = 0.0

    # Pre-compute 2^{max_grade}
    max_val = pow(2, max_grade)

    for r, rel in enumerate(scores):
        # Rank is 1-indexed for the 1/r term
        rank = r + 1

        # Calculate R_r (probability of satisfaction)
        # Using the formula: R = (2^g - 1) / 2^max_g
        R = (pow(2, rel) - 1.0) / max_val

        # Add to ERR
        err += p * (R / rank)

        # Update probability of *continuing* (1 - R)
        p *= (1.0 - R)

    return err


def aggregate_metrics(
    per_query_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Aggregate per-query metrics across all queries.

    Args:
        per_query_metrics: List of metric dicts (one per query)

    Returns:
        Dict with '{metric}_mean' and '{metric}_std' for each metric
    """
    if not per_query_metrics:
        return {}

    all_keys = per_query_metrics[0].keys()
    aggregated = {}

    for key in all_keys:
        values = [m[key] for m in per_query_metrics]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    return aggregated


# ============================================================
# Self-test with known test cases
# ============================================================


def _run_tests():
    """Run self-tests to verify metric correctness."""
    print("=" * 60)
    print("RANKING METRICS SELF-TEST")
    print("=" * 60)

    passed = 0
    total = 0

    def assert_close(actual, expected, name, tol=1e-6):
        nonlocal passed, total
        total += 1
        if abs(actual - expected) < tol:
            print(f"  ✅ {name}: {actual:.6f} == {expected:.6f}")
            passed += 1
        else:
            print(f"  ❌ {name}: {actual:.6f} != {expected:.6f}")

    # --- Test 1: Perfect ranking ---
    print("\nTest 1: Perfect ranking [1, 1, 0, 0]")
    scores = [1.0, 1.0, 0.0, 0.0]
    assert_close(ndcg_at_k(scores, 4), 1.0, "NDCG@4")
    assert_close(reciprocal_rank(scores), 1.0, "MRR")
    assert_close(average_precision(scores), 1.0, "MAP")

    # --- Test 2: Worst ranking (relevant docs at the end) ---
    print("\nTest 2: Reversed ranking [0, 0, 1, 1]")
    scores = [0.0, 0.0, 1.0, 1.0]
    ndcg_val = ndcg_at_k(scores, 4)
    assert_close(reciprocal_rank(scores), 1.0 / 3.0, "MRR")
    print(f"  ℹ️  NDCG@4 = {ndcg_val:.4f} (should be < 1.0)")
    assert ndcg_val < 1.0, "NDCG should be < 1.0 for reversed ranking"
    passed += 1
    total += 1

    # --- Test 3: Single relevant doc at rank 3 ---
    print("\nTest 3: Single relevant at rank 3 [0, 0, 1, 0, 0]")
    scores = [0.0, 0.0, 1.0, 0.0, 0.0]
    assert_close(reciprocal_rank(scores), 1.0 / 3.0, "MRR")
    assert_close(average_precision(scores), 1.0 / 3.0, "MAP")

    # --- Test 4: All relevant ---
    print("\nTest 4: All relevant [1, 1, 1]")
    scores = [1.0, 1.0, 1.0]
    assert_close(ndcg_at_k(scores, 3), 1.0, "NDCG@3")
    assert_close(reciprocal_rank(scores), 1.0, "MRR")
    assert_close(average_precision(scores), 1.0, "MAP")

    # --- Test 5: No relevant ---
    print("\nTest 5: None relevant [0, 0, 0]")
    scores = [0.0, 0.0, 0.0]
    assert_close(ndcg_at_k(scores, 3), 0.0, "NDCG@3")
    assert_close(reciprocal_rank(scores), 0.0, "MRR")
    assert_close(average_precision(scores), 0.0, "MAP")

    # --- Test 6: Graded relevance ---
    print("\nTest 6: Graded relevance [0.8, 0.2, 1.0, 0.0]")
    scores = [0.8, 0.2, 1.0, 0.0]
    ndcg_val = ndcg_at_k(scores, 4)
    print(f"  ℹ️  NDCG@4 = {ndcg_val:.4f} (should be < 1.0, imperfect order)")
    assert 0.0 < ndcg_val < 1.0
    passed += 1
    total += 1

    # --- Test 7: NDCG@k truncation ---
    print("\nTest 7: NDCG@1 for [1, 0, 0] vs [0, 1, 0]")
    assert_close(ndcg_at_k([1.0, 0.0, 0.0], 1), 1.0, "NDCG@1 (relevant first)")
    assert_close(ndcg_at_k([0.0, 1.0, 0.0], 1), 0.0, "NDCG@1 (relevant second)")

    # --- Test 8: compute_ranking_metrics integration ---
    print("\nTest 8: compute_ranking_metrics integration")
    metrics = compute_ranking_metrics([1.0, 0.0, 1.0, 0.0], ks=[1, 3, 5])
    assert "ndcg@1" in metrics and "ndcg@3" in metrics and "ndcg@5" in metrics
    assert "mrr" in metrics and "map" in metrics
    print(f"  ℹ️  Keys: {list(metrics.keys())}")
    passed += 1
    total += 1

    # --- Test 9: aggregate_metrics ---
    print("\nTest 9: aggregate_metrics")
    per_query = [
        {"ndcg@5": 1.0, "mrr": 1.0, "map": 1.0},
        {"ndcg@5": 0.5, "mrr": 0.5, "map": 0.5},
    ]
    agg = aggregate_metrics(per_query)
    assert_close(agg["ndcg@5_mean"], 0.75, "NDCG@5 mean")
    assert_close(agg["mrr_mean"], 0.75, "MRR mean")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"PASSED: {passed}/{total}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _run_tests()
