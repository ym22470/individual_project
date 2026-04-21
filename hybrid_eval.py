"""
Hybrid Search Evaluation: BM25 keyword matching + HNSW cosine similarity
combined via Reciprocal Rank Fusion (RRF).

Evaluates both single-target (benchmark_queries.jsonl) and multi-target
(benchmark_queries_multi.json) benchmarks, reporting pure-cosine vs hybrid
side-by-side for each model.

Outputs:
  /home/chongshengwang/naratix/hybrid_eval_single.png
  /home/chongshengwang/naratix/hybrid_eval_multi.png
  /home/chongshengwang/naratix/hybrid_eval.out
"""

import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import chromadb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── paths ──────────────────────────────────────────────────────────────────
SINGLE_QUERIES_PATH = "/home/chongshengwang/naratix/benchmark_queries.jsonl"
MULTI_QUERIES_PATH  = "/home/chongshengwang/naratix/benchmark_queries_multi.json"
CHROMA_DIR          = "/home/chongshengwang/naratix/benchmark_cache/chromadb_eval"
OUT_TEXT            = "/home/chongshengwang/naratix/hybrid_eval.out"
OUT_IMG_SINGLE      = "/home/chongshengwang/naratix/hybrid_eval_single.png"
OUT_IMG_MULTI       = "/home/chongshengwang/naratix/hybrid_eval_multi.png"

MODELS      = ["base_quantized", "long_quantized", "tags_quantized"]
MODEL_DISP  = {"base_quantized": "Base", "long_quantized": "Long", "tags_quantized": "Tags"}
QUERY_TYPES = ["exact", "paraphrase", "semantic", "motion", "style"]
RRF_K       = 60        # standard RRF constant
K_EVAL      = 5         # Recall@K / Precision@K

COLORS = {
    "cosine":  {"base_quantized": "#4C72B0", "long_quantized": "#DD8452", "tags_quantized": "#55A868"},
    "hybrid":  {"base_quantized": "#1a3a6b", "long_quantized": "#8b4010", "tags_quantized": "#1e6b37"},
}


# ── helpers ────────────────────────────────────────────────────────────────

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


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings: List[List[str]], k: int = RRF_K) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def load_single_queries(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
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
            row = json.loads(line)
            query      = (row.get("query") or "").strip()
            target_ids = [t.strip() for t in (row.get("target_ids") or []) if t.strip()]
            qtype      = (row.get("type") or "unknown").strip()
            if query and target_ids:
                rows.append({"query": query, "target_ids": target_ids, "type": qtype})
    return rows


# ── collection loader ──────────────────────────────────────────────────────

def load_collection(model_label: str):
    path   = os.path.join(CHROMA_DIR, model_label)
    client = chromadb.PersistentClient(path=path)
    cols   = client.list_collections()
    if not cols:
        raise RuntimeError(f"No collection found for {model_label}")
    col    = cols[0]
    data   = col.get(include=["documents"])
    ids    = data["ids"]
    docs   = data["documents"]
    return col, ids, docs


def build_bm25(ids: List[str], docs: List[str]) -> Tuple[BM25Okapi, List[str]]:
    corpus = [tokenize(d) for d in docs]
    return BM25Okapi(corpus), ids


# ── ranking ────────────────────────────────────────────────────────────────

def cosine_rank(collection, embedding_model, query: str, n_docs: int) -> List[str]:
    emb = embedding_model.encode(f"search_query: {query}").tolist()
    res = collection.query(query_embeddings=[emb], n_results=n_docs, include=["distances"])
    return res["ids"][0]


def bm25_rank(bm25: BM25Okapi, bm25_ids: List[str], query: str) -> List[str]:
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    order  = np.argsort(scores)[::-1]
    return [bm25_ids[i] for i in order]


def hybrid_rank(collection, embedding_model, bm25: BM25Okapi,
                bm25_ids: List[str], query: str, n_docs: int) -> List[str]:
    dense  = cosine_rank(collection, embedding_model, query, n_docs)
    sparse = bm25_rank(bm25, bm25_ids, query)
    return rrf_fuse([dense, sparse])


# ── single-target metrics ──────────────────────────────────────────────────

def average_precision_multi(ranked: List[str], relevant: Set[str]) -> float:
    hits, score = 0, 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant) if relevant else 0.0


def eval_single(queries: List[Dict], collection, embedding_model,
                bm25: BM25Okapi, bm25_ids: List[str],
                key_to_id: Dict[str, str], n_docs: int) -> Dict:
    buckets = {
        mode: {qt: {"h1": 0, "h5": 0, "mrr": 0.0, "n": 0} for qt in QUERY_TYPES}
        for mode in ("cosine", "hybrid")
    }

    for row in queries:
        qtype     = row["type"]
        if qtype not in buckets["cosine"]:
            continue
        target_id = key_to_id.get(canonicalize(row["target"]))
        if target_id is None:
            continue

        for mode in ("cosine", "hybrid"):
            if mode == "cosine":
                ranked = cosine_rank(collection, embedding_model, row["query"], n_docs)
            else:
                ranked = hybrid_rank(collection, embedding_model, bm25, bm25_ids, row["query"], n_docs)

            b = buckets[mode][qtype]
            b["n"] += 1
            if ranked[0] == target_id:
                b["h1"] += 1
            if target_id in ranked[:K_EVAL]:
                b["h5"] += 1
            for rank, doc_id in enumerate(ranked, start=1):
                if doc_id == target_id:
                    b["mrr"] += 1.0 / rank
                    break

    result = {}
    for mode, type_buckets in buckets.items():
        result[mode] = {}
        for qt, b in type_buckets.items():
            n = max(b["n"], 1)
            result[mode][qt] = {
                "Recall@1": b["h1"] / n,
                f"Recall@{K_EVAL}": b["h5"] / n,
                "MRR": b["mrr"] / n,
                "n": b["n"],
            }
    return result


# ── multi-target metrics ───────────────────────────────────────────────────

def eval_multi(queries: List[Dict], collection, embedding_model,
               bm25: BM25Okapi, bm25_ids: List[str],
               key_to_id: Dict[str, str], n_docs: int) -> Dict:
    buckets = {
        mode: {qt: {"recall": 0.0, "prec": 0.0, "map": 0.0, "n": 0} for qt in QUERY_TYPES}
        for mode in ("cosine", "hybrid")
    }

    for row in queries:
        qtype = row["type"]
        if qtype not in buckets["cosine"]:
            continue
        relevant = {key_to_id[canonicalize(t)] for t in row["target_ids"] if canonicalize(t) in key_to_id}
        if not relevant:
            continue

        for mode in ("cosine", "hybrid"):
            if mode == "cosine":
                ranked = cosine_rank(collection, embedding_model, row["query"], n_docs)
            else:
                ranked = hybrid_rank(collection, embedding_model, bm25, bm25_ids, row["query"], n_docs)

            top_k  = ranked[:K_EVAL]
            hits_k = sum(1 for d in top_k if d in relevant)
            b      = buckets[mode][qtype]
            b["recall"] += hits_k / len(relevant)
            b["prec"]   += hits_k / K_EVAL
            b["map"]    += average_precision_multi(ranked, relevant)
            b["n"]      += 1

    result = {}
    for mode, type_buckets in buckets.items():
        result[mode] = {}
        for qt, b in type_buckets.items():
            n = max(b["n"], 1)
            result[mode][qt] = {
                f"Recall@{K_EVAL}": b["recall"] / n,
                f"Precision@{K_EVAL}": b["prec"] / n,
                "MAP": b["map"] / n,
                "n": b["n"],
            }
    return result


# ── rendering ──────────────────────────────────────────────────────────────

def render_comparison(all_results: Dict, metrics: List[str], title: str,
                      output_path: str, n_label: str) -> None:
    active_models = [m for m in MODELS if m in all_results]
    n_metrics     = len(metrics)
    bar_width     = 0.13
    x             = np.arange(len(QUERY_TYPES))

    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for col, metric in enumerate(metrics):
        ax = axes[col]
        slot = 0
        for model in active_models:
            for mode in ("cosine", "hybrid"):
                vals   = [all_results[model][mode][qt][metric] for qt in QUERY_TYPES]
                offset = (slot - (len(active_models) * 2 - 1) / 2) * bar_width
                label  = f"{MODEL_DISP[model]} {'(cosine)' if mode == 'cosine' else '(hybrid)'}"
                color  = COLORS[mode][model]
                hatch  = None if mode == "cosine" else "//"
                bars   = ax.bar(x + offset, vals, bar_width,
                                label=label, color=color, alpha=0.85,
                                hatch=hatch, edgecolor="white")
                for bar, v in zip(bars, vals):
                    if v > 0.01:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.005,
                                f"{v:.2f}", ha="center", va="bottom",
                                fontsize=6, rotation=90)
                slot += 1

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([qt.capitalize() for qt in QUERY_TYPES], fontsize=9)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_xlabel("Query Type", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(0.5, -0.02, n_label, ha="center", fontsize=9, color="#555555")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def print_comparison_table(all_results: Dict, metrics: List[str], heading: str) -> None:
    active = [m for m in MODELS if m in all_results]
    col_w, type_w = 10, 12
    print(f"\n{'=' * 80}")
    print(heading)
    print(f"{'=' * 80}")
    for metric in metrics:
        print(f"\n--- {metric} ---")
        header = f"{'Type':<{type_w}}"
        for m in active:
            header += f"  {MODEL_DISP[m]+' cos':>{col_w}}  {MODEL_DISP[m]+' hyb':>{col_w}}"
        print(header)
        print("-" * len(header))
        for qt in QUERY_TYPES:
            row = f"{qt.capitalize():<{type_w}}"
            for m in active:
                cos = all_results[m]["cosine"][qt][metric]
                hyb = all_results[m]["hybrid"][qt][metric]
                delta = hyb - cos
                sign  = "+" if delta >= 0 else ""
                row  += f"  {cos:>{col_w}.3f}  {hyb:>{col_w}.3f}"
            print(row)
        # delta summary row
        row = f"{'Δ hybrid':>{type_w}}"
        for m in active:
            cos_avg = sum(all_results[m]["cosine"][qt][metric] for qt in QUERY_TYPES) / len(QUERY_TYPES)
            hyb_avg = sum(all_results[m]["hybrid"][qt][metric] for qt in QUERY_TYPES) / len(QUERY_TYPES)
            delta   = hyb_avg - cos_avg
            sign    = "+" if delta >= 0 else ""
            row    += f"  {'avg':>{col_w}}  {sign+f'{delta:.3f}':>{col_w}}"
        print(row)


# ── main ───────────────────────────────────────────────────────────────────

def run():
    print("Loading queries...")
    single_queries = load_single_queries(SINGLE_QUERIES_PATH)
    multi_queries  = load_multi_queries(MULTI_QUERIES_PATH)
    print(f"  Single-target: {len(single_queries)} queries")
    print(f"  Multi-target:  {len(multi_queries)} queries")

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

        collection, ids, docs = load_collection(model)
        bm25, bm25_ids        = build_bm25(ids, docs)
        key_to_id             = {canonicalize(i): i for i in ids}
        n_docs                = len(ids)

        print(f"  BM25 index built over {n_docs} documents")

        print("  Running single-target eval...")
        single_results[model] = eval_single(
            single_queries, collection, emb_model, bm25, bm25_ids, key_to_id, n_docs)

        print("  Running multi-target eval...")
        multi_results[model] = eval_multi(
            multi_queries, collection, emb_model, bm25, bm25_ids, key_to_id, n_docs)

    # ── print tables ──
    print_comparison_table(single_results,
        ["Recall@1", f"Recall@{K_EVAL}", "MRR"],
        "SINGLE-TARGET: Cosine vs Hybrid")

    print_comparison_table(multi_results,
        [f"Recall@{K_EVAL}", f"Precision@{K_EVAL}", "MAP"],
        "MULTI-TARGET (Concept Understanding): Cosine vs Hybrid")

    # ── render images ──
    n_single = {qt: single_results[MODELS[0]]["cosine"][qt]["n"] for qt in QUERY_TYPES if MODELS[0] in single_results}
    n_multi  = {qt: multi_results[MODELS[0]]["cosine"][qt]["n"]  for qt in QUERY_TYPES if MODELS[0] in multi_results}

    render_comparison(
        single_results,
        metrics=["Recall@1", "MRR"],
        title="Hybrid vs Cosine — Single-Target Benchmark",
        output_path=OUT_IMG_SINGLE,
        n_label="  |  ".join(f"{qt.capitalize()}: n={n_single.get(qt, 0)}" for qt in QUERY_TYPES),
    )

    render_comparison(
        multi_results,
        metrics=[f"Recall@{K_EVAL}", "MAP"],
        title="Hybrid vs Cosine — Concept Understanding Benchmark (Multi-Target)",
        output_path=OUT_IMG_MULTI,
        n_label="  |  ".join(f"{qt.capitalize()}: n={n_multi.get(qt, 0)}" for qt in QUERY_TYPES),
    )


def main():
    os.makedirs(os.path.dirname(OUT_TEXT), exist_ok=True)
    original_stdout = sys.stdout
    with open(OUT_TEXT, "w", encoding="utf-8") as log_fh:
        sys.stdout = TeeStream(original_stdout, log_fh)
        try:
            run()
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
