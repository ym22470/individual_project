"""
Statistical Significance Tests — Paired Wilcoxon Signed-Rank Test.

Compares per-query scores between model pairs on both benchmarks.
Uses Wilcoxon (non-parametric, no normality assumption) since per-query
scores are bounded [0,1] and distributions are skewed.

Comparisons tested:
  A) Long vs Base   — validates fine-tuning direction
  B) Long vs Tags   — validates description length over conciseness
  C) Long Hybrid vs Long Cosine — validates hybrid search decision

Metrics tested per query type:
  Single-target : NDCG@5, Recall@1, MRR
  Multi-target  : NDCG@5, MAP

Significance levels:
  ***  p < 0.001
  **   p < 0.01
  *    p < 0.05
  ~    p < 0.10  (trend)
  ns   not significant

Outputs:
  /home/chongshengwang/naratix/significance_tests.png
  /home/chongshengwang/naratix/significance_tests.out
"""

import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import chromadb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from rank_bm25 import BM25Okapi
from scipy import stats
from sentence_transformers import SentenceTransformer

SINGLE_QUERIES_PATH = "/home/chongshengwang/naratix/benchmark_queries.jsonl"
MULTI_QUERIES_PATH  = "/home/chongshengwang/naratix/benchmark_queries_multi.json"
CHROMA_DIR          = "/home/chongshengwang/naratix/benchmark_cache/chromadb_eval"
OUT_IMAGE_DIR       = "/home/chongshengwang/naratix"
OUT_TEXT            = "/home/chongshengwang/naratix/significance_tests.out"

MODELS      = ["base_quantized", "long_quantized", "tags_quantized"]
MODEL_DISP  = {"base_quantized": "Base", "long_quantized": "Long", "tags_quantized": "Tags"}
QUERY_TYPES = ["exact", "paraphrase", "semantic", "motion", "style"]
NDCG_K      = 5
RRF_K       = 60


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


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "~"
    return "ns"


# ── metric helpers ────────────────────────────────────────────────────────────

def ndcg_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    def dcg(r, rel, k):
        return sum(1.0 / math.log2(i + 2)
                   for i, d in enumerate(r[:k]) if d in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg(ranked, relevant, k) / ideal if ideal > 0 else 0.0


def mrr(ranked: List[str], relevant: Set[str]) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(ranked: List[str], relevant: Set[str]) -> float:
    hits, score = 0, 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant) if relevant else 0.0


# ── data loading ──────────────────────────────────────────────────────────────

def load_single_queries(path: str) -> List[Dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            query  = (row.get("query") or "").strip()
            target = (row.get("target_id") or row.get("video_id") or "").strip()
            qtype  = (row.get("type") or "unknown").strip()
            if query and target:
                rows.append({"query": query, "target": target, "type": qtype})
    return rows


def load_multi_queries(path: str) -> List[Dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            row        = json.loads(line)
            query      = (row.get("query") or "").strip()
            target_ids = [t.strip() for t in (row.get("target_ids") or []) if t.strip()]
            qtype      = (row.get("type") or "unknown").strip()
            if query and target_ids:
                rows.append({"query": query, "target_ids": target_ids, "type": qtype})
    return rows


def load_collection(model_label: str):
    path   = os.path.join(CHROMA_DIR, model_label)
    client = chromadb.PersistentClient(path=path)
    col    = client.list_collections()[0]
    data   = col.get(include=["documents"])
    ids, docs = data["ids"], data["documents"]
    key_to_id = {canonicalize(i): i for i in ids}
    bm25      = BM25Okapi([tokenize(d) for d in docs])
    return col, ids, key_to_id, bm25


def cosine_rank(col, emb_model, query: str, n: int) -> List[str]:
    emb = emb_model.encode(f"search_query: {query}").tolist()
    return col.query(query_embeddings=[emb], n_results=n, include=["distances"])["ids"][0]


def hybrid_rank(col, emb_model, bm25: BM25Okapi, bm25_ids: List[str],
                query: str, n: int) -> List[str]:
    dense  = cosine_rank(col, emb_model, query, n)
    sparse_scores = bm25.get_scores(tokenize(query))
    sparse = [bm25_ids[i] for i in np.argsort(sparse_scores)[::-1]]
    scores: Dict[str, float] = defaultdict(float)
    for rank, d in enumerate(dense,  start=1): scores[d] += 1.0 / (RRF_K + rank)
    for rank, d in enumerate(sparse, start=1): scores[d] += 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


# ── per-query score collection ────────────────────────────────────────────────

def collect_single_scores(queries, col, emb_model, bm25, ids, key_to_id):
    """Returns {qtype: {metric: [per_query_scores]}} for cosine and hybrid."""
    cosine = {qt: {"ndcg5": [], "recall1": [], "mrr": []} for qt in QUERY_TYPES}
    hybrid = {qt: {"ndcg5": [], "recall1": [], "mrr": []} for qt in QUERY_TYPES}
    n = len(ids)

    for row in queries:
        qt = row["type"]
        if qt not in cosine: continue
        target_id = key_to_id.get(canonicalize(row["target"]))
        if target_id is None: continue
        relevant = {target_id}

        c_ranked = cosine_rank(col, emb_model, row["query"], n)
        h_ranked = hybrid_rank(col, emb_model, bm25, ids, row["query"], n)

        for ranked, bucket in [(c_ranked, cosine), (h_ranked, hybrid)]:
            bucket[qt]["ndcg5"].append(ndcg_at_k(ranked, relevant, NDCG_K))
            bucket[qt]["recall1"].append(1.0 if ranked[0] == target_id else 0.0)
            bucket[qt]["mrr"].append(mrr(ranked, relevant))

    return cosine, hybrid


def collect_multi_scores(queries, col, emb_model, bm25, ids, key_to_id):
    """Returns {qtype: {metric: [per_query_scores]}} for cosine and hybrid."""
    cosine = {qt: {"ndcg5": [], "map": []} for qt in QUERY_TYPES}
    hybrid = {qt: {"ndcg5": [], "map": []} for qt in QUERY_TYPES}
    n = len(ids)

    for row in queries:
        qt = row["type"]
        if qt not in cosine: continue
        relevant = {key_to_id[canonicalize(t)] for t in row["target_ids"]
                    if canonicalize(t) in key_to_id}
        if not relevant: continue

        c_ranked = cosine_rank(col, emb_model, row["query"], n)
        h_ranked = hybrid_rank(col, emb_model, bm25, ids, row["query"], n)

        for ranked, bucket in [(c_ranked, cosine), (h_ranked, hybrid)]:
            bucket[qt]["ndcg5"].append(ndcg_at_k(ranked, relevant, NDCG_K))
            bucket[qt]["map"].append(average_precision(ranked, relevant))

    return cosine, hybrid


# ── Wilcoxon test ─────────────────────────────────────────────────────────────

def wilcoxon_test(a: List[float], b: List[float]) -> Tuple[float, float, float]:
    """Returns (mean_a, mean_b, p_value). Uses alternative='two-sided'."""
    if len(a) < 10 or len(b) < 10:
        return float("nan"), float("nan"), float("nan")
    try:
        _, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        p = 1.0
    return float(np.mean(a)), float(np.mean(b)), p


# ── printing ──────────────────────────────────────────────────────────────────

def print_comparison(name_a: str, name_b: str, scores_a: Dict, scores_b: Dict,
                     metrics: List[str], heading: str) -> List[Tuple]:
    """Prints table and returns rows for heatmap: (label, metric, p, delta)."""
    rows = []
    w = 10
    print(f"\n{'=' * 75}")
    print(f"{heading}: {name_a}  vs  {name_b}")
    print(f"{'=' * 75}")
    header = f"{'Type':<12}{'Metric':<12}{name_a:>{w}}  {name_b:>{w}}  {'Δ':>{w}}  {'p-value':>{w}}  Sig"
    print(header)
    print("-" * len(header))

    for qt in QUERY_TYPES + ["overall"]:
        for metric in metrics:
            if qt == "overall":
                a_vals = [v for t in QUERY_TYPES for v in scores_a[t][metric]]
                b_vals = [v for t in QUERY_TYPES for v in scores_b[t][metric]]
            else:
                a_vals = scores_a[qt][metric]
                b_vals = scores_b[qt][metric]

            mean_a, mean_b, p = wilcoxon_test(a_vals, b_vals)
            if math.isnan(p):
                print(f"  {qt.capitalize():<10}{metric:<12}{'n/a':>{w}}  {'n/a':>{w}}  {'n/a':>{w}}  {'n/a':>{w}}  --")
                continue
            delta = mean_b - mean_a
            sign  = "+" if delta >= 0 else ""
            stars = sig_stars(p)
            print(f"  {qt.capitalize():<10}{metric:<12}{mean_a:>{w}.3f}  {mean_b:>{w}.3f}  "
                  f"{sign+f'{delta:.3f}':>{w}}  {p:>{w}.4f}  {stars}")
            rows.append((f"{qt.capitalize()}\n{metric}", mean_b - mean_a, p, stars))

    return rows


# ── heatmap rendering ─────────────────────────────────────────────────────────

def render_single_panel(title: str, rows: List, output_path: str) -> None:
    """Save one comparison panel as its own figure."""
    from matplotlib.patches import Patch
    import math as _math

    if not rows:
        return

    cmap   = plt.cm.RdYlGn
    norm   = mcolors.Normalize(vmin=0, vmax=4)
    labels = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    pvals  = [r[2] for r in rows]
    stars  = [r[3] for r in rows]

    log_p  = [-_math.log10(max(p, 1e-6)) for p in pvals]
    colors = [cmap(norm(lp)) for lp in log_p]
    y_pos  = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.45)))
    fig.suptitle("Statistical Significance — Paired Wilcoxon Signed-Rank Test\n"
                 "(bar = Δ mean score;  colour = -log₁₀(p);  border = significance level)",
                 fontsize=11, y=1.02)

    bars = ax.barh(y_pos, deltas, color=colors, edgecolor="white", height=0.7)

    for bar, star in zip(bars, stars):
        if star in ("***", "**", "*"):
            bar.set_edgecolor("#222222")
            bar.set_linewidth(2.0)
        elif star == "~":
            bar.set_edgecolor("#666666")
            bar.set_linewidth(1.2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ Mean Score (B − A)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.margins(x=0.22)
    ax.autoscale_view()
    x_min, x_max = ax.get_xlim()
    offset = (x_max - x_min) * 0.03

    for bar, delta, star in zip(bars, deltas, stars):
        x_text = (delta + offset) if delta >= 0 else (delta - offset)
        ha     = "left" if delta >= 0 else "right"
        sign   = "+" if delta >= 0 else ""
        ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                f"{sign}{delta:.3f} {star}",
                va="center", ha=ha, fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label("-log₁₀(p)", fontsize=9)
    cb.set_ticks([0, 1, 2, 3, 4])
    cb.set_ticklabels(["1.0", "0.1", "0.01", "0.001", "≤0.0001"])

    legend_elements = [
        Patch(facecolor="white", edgecolor="#222222", linewidth=2,   label="*** p<0.001 / ** p<0.01 / * p<0.05"),
        Patch(facecolor="white", edgecolor="#666666", linewidth=1.2, label="~  p<0.10 (trend)"),
        Patch(facecolor="white", edgecolor="white",   linewidth=0.5, label="ns  not significant"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def render_heatmap(comparisons: List[Tuple[str, List]], output_dir: str) -> None:
    """Save each comparison panel as a separate PNG file."""
    filenames = [
        "sig_A_long_vs_base_single.png",
        "sig_A_long_vs_base_multi.png",
        "sig_B_long_vs_tags_single.png",
        "sig_C_hybrid_vs_cosine_single.png",
        "sig_C_hybrid_vs_cosine_multi.png",
    ]
    for (title, rows), fname in zip(comparisons, filenames):
        render_single_panel(title, rows, os.path.join(output_dir, fname))


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    print("Loading queries...")
    single_queries = load_single_queries(SINGLE_QUERIES_PATH)
    multi_queries  = load_multi_queries(MULTI_QUERIES_PATH)

    print("Loading embedding model...")
    emb_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    # collect per-query scores for every model
    single_cosine, single_hybrid, multi_cosine, multi_hybrid = {}, {}, {}, {}

    for model in MODELS:
        model_path = os.path.join(CHROMA_DIR, model)
        if not os.path.isdir(model_path):
            print(f"Skipping {model}")
            continue
        print(f"\nCollecting scores for {model}...")
        col, ids, key_to_id, bm25 = load_collection(model)
        sc, sh = collect_single_scores(single_queries, col, emb_model, bm25, ids, key_to_id)
        mc, mh = collect_multi_scores(multi_queries,  col, emb_model, bm25, ids, key_to_id)
        single_cosine[model] = sc
        single_hybrid[model] = sh
        multi_cosine[model]  = mc
        multi_hybrid[model]  = mh

    heatmap_data = []

    # ── A) Long vs Base (single-target) ──
    rows_A_s = print_comparison(
        "Base (cosine)", "Long (cosine)",
        single_cosine["base_quantized"], single_cosine["long_quantized"],
        ["ndcg5", "recall1", "mrr"],
        "SINGLE-TARGET  A) Long vs Base")
    heatmap_data.append(("A) Long vs Base\nSingle-Target", rows_A_s))

    # ── A) Long vs Base (multi-target) ──
    rows_A_m = print_comparison(
        "Base (cosine)", "Long (cosine)",
        multi_cosine["base_quantized"], multi_cosine["long_quantized"],
        ["ndcg5", "map"],
        "MULTI-TARGET  A) Long vs Base")
    heatmap_data.append(("A) Long vs Base\nMulti-Target", rows_A_m))

    # ── B) Long vs Tags (single-target) ──
    rows_B_s = print_comparison(
        "Tags (cosine)", "Long (cosine)",
        single_cosine["tags_quantized"], single_cosine["long_quantized"],
        ["ndcg5", "recall1", "mrr"],
        "SINGLE-TARGET  B) Long vs Tags")
    heatmap_data.append(("B) Long vs Tags\nSingle-Target", rows_B_s))

    # ── C) Long Hybrid vs Long Cosine (single-target) ──
    rows_C_s = print_comparison(
        "Long (cosine)", "Long (hybrid)",
        single_cosine["long_quantized"], single_hybrid["long_quantized"],
        ["ndcg5", "recall1", "mrr"],
        "SINGLE-TARGET  C) Long Hybrid vs Long Cosine")
    heatmap_data.append(("C) Long Hybrid vs Cosine\nSingle-Target", rows_C_s))

    # ── C) Long Hybrid vs Long Cosine (multi-target) ──
    rows_C_m = print_comparison(
        "Long (cosine)", "Long (hybrid)",
        multi_cosine["long_quantized"], multi_hybrid["long_quantized"],
        ["ndcg5", "map"],
        "MULTI-TARGET  C) Long Hybrid vs Long Cosine")
    heatmap_data.append(("C) Long Hybrid vs Cosine\nMulti-Target", rows_C_m))

    render_heatmap(heatmap_data, OUT_IMAGE_DIR)
    print(f"Saved text to {OUT_TEXT}")


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
