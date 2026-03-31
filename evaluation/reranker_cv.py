"""
5-Fold Cross Validation for Neural Reranker Models.

Compares BM25, semantic search, simple hybrid sum, XNet (Cauchy-based),
and KAN (B-spline) rerankers on ranking quality metrics (NDCG@5, MRR, MAP).

Usage:
    python -m evaluation.reranker_cv

Thesis reference: §2.3.2 (Neural rerankers — KAN & XNet)
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

# Ensure project root is on path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.research.kan_model import KAN
from src.research.xnet_model import XNet
from src.research.ranking_metrics import compute_ranking_metrics, aggregate_metrics

# Configuration
DATA_PATH = os.path.join(project_root, "evaluation", "training_data_balanced.csv")
FEATURE_COLUMNS = [
    "semantic_score",
    "bm25_score",
    "title_overlap",
    "category_match",
    "chunk_position",
    "doc_length",
]
RANKING_KS = [1, 3, 5, 10]
SEED = 42

print("## 1. Loading Data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples, {df['query_id'].nunique()} unique queries.")

# 5-Fold CV Setup
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
query_ids = df["query_id"].unique()

all_metrics_flat = []

print("\n## 2. Running 5-Fold Cross Validation...")


def train_model(model, X, y, epochs=100, lr=0.01):
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


fold = 0
for train_idx, test_idx in kf.split(query_ids):
    fold += 1

    train_qids = query_ids[train_idx]
    test_qids = query_ids[test_idx]

    train_df = df[df["query_id"].isin(train_qids)]
    test_df = df[df["query_id"].isin(test_qids)]

    # Prepare training data
    X_train = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.float32)

    # Train Models
    kan = KAN(layers_hidden=[6, 8, 1])
    kan = train_model(kan, X_train, y_train)

    xnet = XNet(input_dim=6)
    xnet = train_model(xnet, X_train, y_train)

    # Evaluate on Test Fold
    def evaluate_fold(model, df, qids, model_type="pytorch"):
        fold_res = []
        for qid in qids:
            qdata = df[df["query_id"] == qid].copy()
            if len(qdata) < 2:
                continue

            relevance = qdata["relevance_score"].values

            if model_type == "bm25":
                scores = qdata["bm25_score"].values
            elif model_type == "semantic":
                scores = qdata["semantic_score"].values
            elif model_type == "simplesum":
                bm25 = qdata["bm25_score"]
                if bm25.max() > bm25.min():
                    bm25_norm = (bm25 - bm25.min()) / (bm25.max() - bm25.min())
                else:
                    bm25_norm = bm25 * 0.0
                scores = bm25_norm + qdata["semantic_score"]
            elif model_type == "pytorch":
                X_q = torch.FloatTensor(
                    qdata[FEATURE_COLUMNS].values.astype(np.float32)
                )
                with torch.no_grad():
                    scores = model(X_q).squeeze().numpy()

            ranking_indices = np.argsort(-scores)
            ordered_relevance = relevance[ranking_indices].tolist()
            fold_res.append(compute_ranking_metrics(ordered_relevance, RANKING_KS))
        return fold_res

    # Collect metrics
    all_metrics_flat.extend(
        [
            {"Model": "BM25 (Baseline)", **m}
            for m in evaluate_fold(None, test_df, test_qids, "bm25")
        ]
    )
    all_metrics_flat.extend(
        [
            {"Model": "Semantic (Baseline)", **m}
            for m in evaluate_fold(None, test_df, test_qids, "semantic")
        ]
    )
    all_metrics_flat.extend(
        [
            {"Model": "Simple Sum (Hybrid)", **m}
            for m in evaluate_fold(None, test_df, test_qids, "simplesum")
        ]
    )
    all_metrics_flat.extend(
        [
            {"Model": "XNet (Filter)", **m}
            for m in evaluate_fold(xnet, test_df, test_qids, "pytorch")
        ]
    )
    all_metrics_flat.extend(
        [
            {"Model": "KAN (Learned)", **m}
            for m in evaluate_fold(kan, test_df, test_qids, "pytorch")
        ]
    )

print("5-Fold CV Complete ✅")

# Aggregate
metrics_df = pd.DataFrame(all_metrics_flat)
summary = metrics_df.groupby("Model").mean()[["ndcg@5", "mrr", "map"]].reset_index()
summary.columns = ["Model", "NDCG@5", "MRR", "MAP"]

print(f"\nFINAL RESULTS (Average over 5 Folds / {df['query_id'].nunique()} Queries):")
print(summary.sort_values("NDCG@5", ascending=False).to_markdown(index=False))
