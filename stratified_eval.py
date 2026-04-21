"""
Stratified Evaluation: computes Recall@1, Recall@5, and MRR per query type
(exact, paraphrase, semantic, motion, style) for each model, using the
pre-built ChromaDB collections from the previous benchmark run.

Outputs: /home/chongshengwang/naratix/image.png
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List

import chromadb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_QUERIES_PATH = "/home/chongshengwang/naratix/benchmark_queries.jsonl"
DEFAULT_CHROMA_DIR = "/home/chongshengwang/naratix/benchmark_cache/chromadb_eval"
DEFAULT_OUTPUT_IMAGE = "/home/chongshengwang/naratix/image.png"
MODELS = ["base_quantized", "long_quantized", "tags_quantized"]
QUERY_TYPES = ["exact", "paraphrase", "semantic", "motion", "style"]
METRICS = ["Recall@1", "Recall@5", "MRR"]


def canonicalize_video_key(value: str) -> str:
    stem = os.path.splitext(os.path.basename(value or ""))[0].lower()
    stem = re.sub(r"[^\w]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"(?:^|_)(?:artlist|artist)(?=_|$)", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")


def load_queries(jsonl_path: str) -> List[Dict]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = (row.get("query") or "").strip()
            target = (
                row.get("target_id") or row.get("video_id") or row.get("file_name")
                or row.get("video") or row.get("path") or ""
            ).strip()
            qtype = (row.get("type") or "unknown").strip()
            if query and target:
                rows.append({"query": query, "target": target, "type": qtype})
    return rows


def evaluate_model(
    model_label: str,
    chroma_dir: str,
    queries: List[Dict],
    embedding_model,
) -> Dict[str, Dict[str, float]]:
    model_chroma_path = os.path.join(chroma_dir, model_label)
    client = chromadb.PersistentClient(path=model_chroma_path)
    collections = client.list_collections()
    if not collections:
        raise RuntimeError(f"No ChromaDB collection found for {model_label} at {model_chroma_path}")
    collection = collections[0]
    n_docs = collection.count()

    all_ids = collection.get(include=[])["ids"]
    key_to_id = {canonicalize_video_key(doc_id): doc_id for doc_id in all_ids}

    type_buckets: Dict[str, Dict] = {
        qt: {"hits1": 0, "hits5": 0, "mrr": 0.0, "n": 0}
        for qt in QUERY_TYPES
    }

    for row in queries:
        qtype = row["type"]
        if qtype not in type_buckets:
            continue

        target_key = canonicalize_video_key(row["target"])
        target_id = key_to_id.get(target_key)
        if target_id is None:
            continue

        query_emb = embedding_model.encode(f"search_query: {row['query']}").tolist()
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=n_docs,
            include=["distances"],
        )
        ranked_ids = results["ids"][0]

        bucket = type_buckets[qtype]
        bucket["n"] += 1
        if ranked_ids[0] == target_id:
            bucket["hits1"] += 1
        if target_id in ranked_ids[:5]:
            bucket["hits5"] += 1
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id == target_id:
                bucket["mrr"] += 1.0 / rank
                break

    result: Dict[str, Dict[str, float]] = {}
    for qt, b in type_buckets.items():
        n = max(b["n"], 1)
        result[qt] = {
            "Recall@1": b["hits1"] / n,
            "Recall@5": b["hits5"] / n,
            "MRR": b["mrr"] / n,
            "n": b["n"],
        }
    return result


def render_image(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str,
) -> None:
    n_metrics = len(METRICS)
    n_types = len(QUERY_TYPES)
    n_models = len(MODELS)
    bar_width = 0.22
    x = np.arange(n_types)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    model_display = {
        "base_quantized": "Base",
        "long_quantized": "Long",
        "tags_quantized": "Tags",
    }

    fig, axes = plt.subplots(1, n_metrics, figsize=(18, 6))
    fig.suptitle("Stratified Evaluation by Query Type", fontsize=16, fontweight="bold", y=1.01)

    for col, metric in enumerate(METRICS):
        ax = axes[col]
        for m_idx, model in enumerate(MODELS):
            vals = [
                all_results[model][qt][metric]
                for qt in QUERY_TYPES
            ]
            offset = (m_idx - 1) * bar_width
            bars = ax.bar(x + offset, vals, bar_width, label=model_display[model], color=colors[m_idx], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=0,
                )

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([qt.capitalize() for qt in QUERY_TYPES], fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_xlabel("Query Type", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    n_per_type = {qt: all_results[MODELS[0]][qt]["n"] for qt in QUERY_TYPES}
    subtitle = "  |  ".join(f"{qt.capitalize()}: n={n_per_type[qt]}" for qt in QUERY_TYPES)
    fig.text(0.5, -0.02, subtitle, ha="center", fontsize=9, color="#555555")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved image to {output_path}")


def print_table(all_results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    col_w = 14
    type_w = 12
    print("\n=== Stratified Evaluation ===")
    for metric in METRICS:
        print(f"\n--- {metric} ---")
        header = f"{'Type':<{type_w}}" + "".join(f"{m:>{col_w}}" for m in MODELS)
        print(header)
        print("-" * len(header))
        for qt in QUERY_TYPES:
            row = f"{qt.capitalize():<{type_w}}"
            for model in MODELS:
                row += f"{all_results[model][qt][metric]:>{col_w}.3f}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description="Stratified evaluation using pre-built ChromaDB collections")
    parser.add_argument("--queries-file", default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--output-image", default=DEFAULT_OUTPUT_IMAGE)
    args = parser.parse_args()

    print("Loading queries...")
    queries = load_queries(args.queries_file)
    print(f"  Loaded {len(queries)} queries")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in MODELS:
        model_path = os.path.join(args.chroma_dir, model)
        if not os.path.isdir(model_path):
            print(f"Skipping {model}: ChromaDB not found at {model_path}")
            continue
        print(f"Evaluating {model}...")
        all_results[model] = evaluate_model(model, args.chroma_dir, queries, embedding_model)

    if not all_results:
        raise RuntimeError("No model results were produced. Run gradio_demo_test_chromadb.py first.")

    print_table(all_results)
    render_image(all_results, args.output_image)


if __name__ == "__main__":
    main()
