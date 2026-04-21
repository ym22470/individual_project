"""
Concept Understanding Evaluation (multi-target queries).

Uses pre-built ChromaDB collections. For each query with multiple target
videos the following metrics are computed per query type:

  Recall@K   – fraction of relevant targets found in the top-K results
  Precision@K – fraction of top-K results that are relevant
  MAP        – mean average precision across all relevant targets

Outputs:
  /home/chongshengwang/naratix/concept_eval.png   – stratified bar charts
  /home/chongshengwang/naratix/concept_eval.out   – plain-text table
"""

import json
import os
import re
import sys
from typing import Dict, List

import chromadb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_QUERIES_PATH = "/home/chongshengwang/naratix/benchmark_queries_multi.json"
DEFAULT_CHROMA_DIR = "/home/chongshengwang/naratix/benchmark_cache/chromadb_eval"
DEFAULT_OUTPUT_IMAGE = "/home/chongshengwang/naratix/concept_eval.png"
DEFAULT_OUTPUT_TEXT = "/home/chongshengwang/naratix/concept_eval.out"

MODELS = ["base_quantized", "long_quantized", "tags_quantized"]
MODEL_DISPLAY = {"base_quantized": "Base", "long_quantized": "Long", "tags_quantized": "Tags"}
QUERY_TYPES = ["exact", "paraphrase", "semantic", "motion", "style"]
K_RECALL = 5
K_PRECISION = 5
COLORS = {"base_quantized": "#4C72B0", "long_quantized": "#DD8452", "tags_quantized": "#55A868"}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def canonicalize(value: str) -> str:
    stem = os.path.splitext(os.path.basename(value or ""))[0].lower()
    stem = re.sub(r"[^\w]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"(?:^|_)(?:artlist|artist)(?=_|$)", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")


def load_queries(path: str) -> List[Dict]:
    queries = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = (row.get("query") or "").strip()
            target_ids = [t.strip() for t in (row.get("target_ids") or []) if t.strip()]
            qtype = (row.get("type") or "unknown").strip()
            if query and target_ids:
                queries.append({"query": query, "target_ids": target_ids, "type": qtype})
    return queries


def average_precision(ranked_ids: List[str], relevant: set) -> float:
    hits, score = 0, 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant) if relevant else 0.0


def evaluate_model(
    model_label: str,
    chroma_dir: str,
    queries: List[Dict],
    embedding_model,
) -> Dict[str, Dict[str, float]]:
    path = os.path.join(chroma_dir, model_label)
    client = chromadb.PersistentClient(path=path)
    collections = client.list_collections()
    if not collections:
        raise RuntimeError(f"No ChromaDB collection for {model_label} at {path}")
    collection = collections[0]
    n_docs = collection.count()

    all_ids = collection.get(include=[])["ids"]
    key_to_id = {canonicalize(doc_id): doc_id for doc_id in all_ids}

    type_buckets: Dict[str, Dict] = {
        qt: {"recall_k": 0.0, "precision_k": 0.0, "map": 0.0, "n": 0}
        for qt in QUERY_TYPES
    }

    print(f"\n{'Query':<55} {'Type':<12} {'R@{K}':<8} {'P@{K}':<8} {'AP':<6}".format(K=K_RECALL))
    print("-" * 90)

    for row in queries:
        qtype = row["type"]
        if qtype not in type_buckets:
            continue

        relevant_ids = set()
        for t in row["target_ids"]:
            doc_id = key_to_id.get(canonicalize(t))
            if doc_id:
                relevant_ids.add(doc_id)
        if not relevant_ids:
            continue

        query_emb = embedding_model.encode(f"search_query: {row['query']}").tolist()
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=n_docs,
            include=["distances"],
        )
        ranked_ids = results["ids"][0]

        top_k = ranked_ids[:K_RECALL]
        hits_in_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        recall_k = hits_in_k / len(relevant_ids)
        precision_k = hits_in_k / K_PRECISION
        ap = average_precision(ranked_ids, relevant_ids)

        bucket = type_buckets[qtype]
        bucket["recall_k"] += recall_k
        bucket["precision_k"] += precision_k
        bucket["map"] += ap
        bucket["n"] += 1

        print(f"{row['query'][:53]:<55} {qtype:<12} {recall_k:<8.3f} {precision_k:<8.3f} {ap:<6.3f}")

    result: Dict[str, Dict[str, float]] = {}
    for qt, b in type_buckets.items():
        n = max(b["n"], 1)
        result[qt] = {
            f"Recall@{K_RECALL}": b["recall_k"] / n,
            f"Precision@{K_PRECISION}": b["precision_k"] / n,
            "MAP": b["map"] / n,
            "n": b["n"],
        }
    return result


def print_table(all_results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    metrics = [f"Recall@{K_RECALL}", f"Precision@{K_PRECISION}", "MAP"]
    col_w, type_w = 16, 12
    print("\n=== Concept Understanding Evaluation (multi-target) ===")
    for metric in metrics:
        print(f"\n--- {metric} ---")
        header = f"{'Type':<{type_w}}" + "".join(f"{MODEL_DISPLAY[m]:>{col_w}}" for m in MODELS if m in all_results)
        print(header)
        print("-" * len(header))
        for qt in QUERY_TYPES:
            row = f"{qt.capitalize():<{type_w}}"
            for m in MODELS:
                if m not in all_results:
                    continue
                row += f"{all_results[m][qt][metric]:>{col_w}.3f}"
            print(row)


def render_image(all_results: Dict[str, Dict[str, Dict[str, float]]], output_path: str) -> None:
    active_models = [m for m in MODELS if m in all_results]
    metrics = [f"Recall@{K_RECALL}", f"Precision@{K_PRECISION}", "MAP"]
    bar_width = 0.22
    x = np.arange(len(QUERY_TYPES))
    colors = [COLORS[m] for m in active_models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Concept Understanding Evaluation by Query Type\n(Multi-target queries)", fontsize=15, fontweight="bold", y=1.02)

    for col, metric in enumerate(metrics):
        ax = axes[col]
        for m_idx, model in enumerate(active_models):
            vals = [all_results[model][qt][metric] for qt in QUERY_TYPES]
            offset = (m_idx - (len(active_models) - 1) / 2) * bar_width
            bars = ax.bar(x + offset, vals, bar_width, label=MODEL_DISPLAY[model], color=colors[m_idx], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([qt.capitalize() for qt in QUERY_TYPES], fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_xlabel("Query Type", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    n_per_type = {qt: all_results[active_models[0]][qt]["n"] for qt in QUERY_TYPES}
    subtitle = "  |  ".join(f"{qt.capitalize()}: n={n_per_type[qt]}" for qt in QUERY_TYPES)
    fig.text(0.5, -0.02, subtitle, ha="center", fontsize=9, color="#555555")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved image to {output_path}")


def main():
    os.makedirs(os.path.dirname(DEFAULT_OUTPUT_TEXT), exist_ok=True)
    original_stdout = sys.stdout
    with open(DEFAULT_OUTPUT_TEXT, "w", encoding="utf-8") as log_fh:
        sys.stdout = TeeStream(original_stdout, log_fh)
        try:
            run()
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout


def run():
    print("Loading queries...")
    queries = load_queries(DEFAULT_QUERIES_PATH)
    print(f"  Loaded {len(queries)} multi-target queries")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in MODELS:
        model_path = os.path.join(DEFAULT_CHROMA_DIR, model)
        if not os.path.isdir(model_path):
            print(f"Skipping {model}: ChromaDB not found at {model_path}")
            continue
        print(f"\n{'=' * 60}")
        print(f"Evaluating {model}")
        print(f"{'=' * 60}")
        all_results[model] = evaluate_model(model, DEFAULT_CHROMA_DIR, queries, embedding_model)

    print_table(all_results)
    render_image(all_results, DEFAULT_OUTPUT_IMAGE)
    print(f"Saved text results to {DEFAULT_OUTPUT_TEXT}")


if __name__ == "__main__":
    main()
