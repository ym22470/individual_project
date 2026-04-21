"""
NDCG@K Evaluation — stratified by query type for all three models.

Uses pre-built ChromaDB collections. Evaluates both single-target
(benchmark_queries.jsonl) and multi-target (benchmark_queries_multi.json)
benchmarks with binary relevance.

  Single-target : one relevant doc per query  → NDCG@K = 1/log2(rank+1)
  Multi-target  : 2-4 relevant docs per query → NDCG captures rank quality
                  of ALL relevant docs, not just whether they appear

Outputs:
  /home/chongshengwang/naratix/ndcg_eval.png
  /home/chongshengwang/naratix/ndcg_eval.out
"""

import json
import math
import os
import re
import sys
from typing import Dict, List, Set

import chromadb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

SINGLE_QUERIES_PATH = "/home/chongshengwang/naratix/benchmark_queries.jsonl"
MULTI_QUERIES_PATH  = "/home/chongshengwang/naratix/benchmark_queries_multi.json"
CHROMA_DIR          = "/home/chongshengwang/naratix/benchmark_cache/chromadb_eval"
OUT_IMAGE           = "/home/chongshengwang/naratix/ndcg_eval.png"
OUT_TEXT            = "/home/chongshengwang/naratix/ndcg_eval.out"

MODELS      = ["base_quantized", "long_quantized", "tags_quantized"]
MODEL_DISP  = {"base_quantized": "Base", "long_quantized": "Long", "tags_quantized": "Tags"}
QUERY_TYPES = ["exact", "paraphrase", "semantic", "motion", "style"]
K_VALUES    = [1, 3, 5, 10]
COLORS      = {"base_quantized": "#4C72B0", "long_quantized": "#DD8452", "tags_quantized": "#55A868"}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
        return len(data)
    def flush(self):
        for s in self.streams: s.flush()


def canonicalize(value: str) -> str:
    stem = os.path.splitext(os.path.basename(value or ""))[0].lower()
    stem = re.sub(r"[^\w]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"(?:^|_)(?:artlist|artist)(?=_|$)", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")


def load_single_queries(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row    = json.loads(line)
            query  = (row.get("query") or "").strip()
            target = (row.get("target_id") or row.get("video_id") or "").strip()
            qtype  = (row.get("type") or "unknown").strip()
            if query and target:
                rows.append({"query": query, "target": target, "type": qtype})
    return rows


def load_multi_queries(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row        = json.loads(line)
            query      = (row.get("query") or "").strip()
            target_ids = [t.strip() for t in (row.get("target_ids") or []) if t.strip()]
            qtype      = (row.get("type") or "unknown").strip()
            if query and target_ids:
                rows.append({"query": query, "target_ids": target_ids, "type": qtype})
    return rows


# ── NDCG formula ────────────────────────────────────────────────────────────

def dcg(ranked: List[str], relevant: Set[str], k: int) -> float:
    score = 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            score += 1.0 / math.log2(rank + 1)
    return score


def idcg(relevant: Set[str], k: int) -> float:
    """Ideal DCG: all relevant docs ranked first."""
    n = min(len(relevant), k)
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, n + 1))


def ndcg_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    ideal = idcg(relevant, k)
    return dcg(ranked, relevant, k) / ideal if ideal > 0 else 0.0


# ── evaluation ───────────────────────────────────────────────────────────────

def get_ranked(collection, embedding_model, query: str, n_docs: int) -> List[str]:
    emb = embedding_model.encode(f"search_query: {query}").tolist()
    res = collection.query(query_embeddings=[emb], n_results=n_docs, include=["distances"])
    return res["ids"][0]


def evaluate_single(queries: List[Dict], collection, embedding_model,
                    key_to_id: Dict[str, str], n_docs: int) -> Dict:
    """Returns {qt: {ndcg@k: score}} averaged over queries."""
    buckets = {qt: {k: [] for k in K_VALUES} for qt in QUERY_TYPES}

    for row in queries:
        qtype = row["type"]
        if qtype not in buckets:
            continue
        target_id = key_to_id.get(canonicalize(row["target"]))
        if target_id is None:
            continue

        ranked   = get_ranked(collection, embedding_model, row["query"], n_docs)
        relevant = {target_id}
        for k in K_VALUES:
            buckets[qtype][k].append(ndcg_at_k(ranked, relevant, k))

    return {
        qt: {k: (sum(v) / len(v) if v else 0.0) for k, v in ks.items()}
        for qt, ks in buckets.items()
    }


def evaluate_multi(queries: List[Dict], collection, embedding_model,
                   key_to_id: Dict[str, str], n_docs: int) -> Dict:
    """Returns {qt: {ndcg@k: score}} averaged over queries."""
    buckets = {qt: {k: [] for k in K_VALUES} for qt in QUERY_TYPES}

    for row in queries:
        qtype = row["type"]
        if qtype not in buckets:
            continue
        relevant = {key_to_id[canonicalize(t)] for t in row["target_ids"]
                    if canonicalize(t) in key_to_id}
        if not relevant:
            continue

        ranked = get_ranked(collection, embedding_model, row["query"], n_docs)
        for k in K_VALUES:
            buckets[qtype][k].append(ndcg_at_k(ranked, relevant, k))

    return {
        qt: {k: (sum(v) / len(v) if v else 0.0) for k, v in ks.items()}
        for qt, ks in buckets.items()
    }


# ── printing ─────────────────────────────────────────────────────────────────

def print_table(all_results: Dict, k: int, heading: str) -> None:
    active   = [m for m in MODELS if m in all_results]
    col_w, type_w = 14, 12
    print(f"\n{'=' * 70}")
    print(f"{heading}  —  NDCG@{k}")
    print(f"{'=' * 70}")
    header = f"{'Type':<{type_w}}" + "".join(f"{MODEL_DISP[m]:>{col_w}}" for m in active)
    print(header)
    print("-" * len(header))
    for qt in QUERY_TYPES:
        row = f"{qt.capitalize():<{type_w}}"
        for m in active:
            row += f"{all_results[m][qt][k]:>{col_w}.3f}"
        print(row)
    # overall average
    row = f"{'Overall':<{type_w}}"
    for m in active:
        avg = sum(all_results[m][qt][k] for qt in QUERY_TYPES) / len(QUERY_TYPES)
        row += f"{avg:>{col_w}.3f}"
    print(row)


# ── rendering ─────────────────────────────────────────────────────────────────

def render(single_results: Dict, multi_results: Dict, output_path: str) -> None:
    active     = [m for m in MODELS if m in single_results]
    plot_k     = [1, 5, 10]          # K values to plot
    bar_width  = 0.22
    x          = np.arange(len(QUERY_TYPES))
    colors     = [COLORS[m] for m in active]

    fig, axes = plt.subplots(2, len(plot_k), figsize=(6 * len(plot_k), 11))
    fig.suptitle("NDCG@K — Single-Target (top) vs Multi-Target (bottom)",
                 fontsize=14, fontweight="bold", y=1.01)

    for row_idx, (results, row_label) in enumerate(
            [(single_results, "Single-Target"), (multi_results, "Multi-Target")]):
        for col_idx, k in enumerate(plot_k):
            ax = axes[row_idx][col_idx]
            for m_idx, model in enumerate(active):
                vals   = [results[model][qt][k] for qt in QUERY_TYPES]
                offset = (m_idx - (len(active) - 1) / 2) * bar_width
                bars   = ax.bar(x + offset, vals, bar_width,
                                label=MODEL_DISP[model],
                                color=colors[m_idx], alpha=0.85)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

            ax.set_title(f"{row_label} — NDCG@{k}", fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([qt.capitalize() for qt in QUERY_TYPES], fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("NDCG", fontsize=9)
            ax.set_xlabel("Query Type", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved image to {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    print("Loading queries...")
    single_queries = load_single_queries(SINGLE_QUERIES_PATH)
    multi_queries  = load_multi_queries(MULTI_QUERIES_PATH)
    print(f"  Single-target : {len(single_queries)} queries")
    print(f"  Multi-target  : {len(multi_queries)} queries")

    print("Loading embedding model...")
    emb_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    single_results: Dict = {}
    multi_results:  Dict = {}

    for model in MODELS:
        model_path = os.path.join(CHROMA_DIR, model)
        if not os.path.isdir(model_path):
            print(f"Skipping {model}: no ChromaDB at {model_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating {model}")
        print(f"{'=' * 60}")

        client     = chromadb.PersistentClient(path=model_path)
        collection = client.list_collections()[0]
        all_ids    = collection.get(include=[])["ids"]
        key_to_id  = {canonicalize(i): i for i in all_ids}
        n_docs     = len(all_ids)

        print("  Running single-target NDCG...")
        single_results[model] = evaluate_single(
            single_queries, collection, emb_model, key_to_id, n_docs)

        print("  Running multi-target NDCG...")
        multi_results[model] = evaluate_multi(
            multi_queries, collection, emb_model, key_to_id, n_docs)

    for k in K_VALUES:
        print_table(single_results, k, "SINGLE-TARGET")
    for k in K_VALUES:
        print_table(multi_results, k, "MULTI-TARGET")

    render(single_results, multi_results, OUT_IMAGE)
    print(f"Saved text results to {OUT_TEXT}")


def main():
    with open(OUT_TEXT, "w", encoding="utf-8") as log_fh:
        original = sys.stdout
        sys.stdout = TeeStream(original, log_fh)
        try:
            run()
        finally:
            sys.stdout.flush()
            sys.stdout = original


if __name__ == "__main__":
    main()
