"""
Latency comparison diagram across the three models.
Reads aggregate query latency from results.out and per-video inference
latency from eval_videos_long_outputs.json (long model only).

Outputs: /home/chongshengwang/naratix/latency.png
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "/home/chongshengwang/naratix/results.out"
LONG_OUTPUTS_PATH = "/home/chongshengwang/naratix/benchmark_cache/eval_videos_long_outputs.json"
OUTPUT_IMAGE = "/home/chongshengwang/naratix/latency.png"

MODEL_LABELS = ["base_quantized", "long_quantized", "tags_quantized"]
MODEL_DISPLAY = {"base_quantized": "Base", "long_quantized": "Long", "tags_quantized": "Tags"}
COLORS = {"base_quantized": "#4C72B0", "long_quantized": "#DD8452", "tags_quantized": "#55A868"}


def parse_query_latencies(results_path: str) -> dict:
    latencies = {}
    with open(results_path) as f:
        text = f.read()
    for model in MODEL_LABELS:
        m = re.search(
            rf"=== Aggregate Metrics \({re.escape(model)}\) ===.*?"
            rf"Cosine avg latency/query:\s+([\d.]+)\s+ms",
            text, re.DOTALL
        )
        if m:
            latencies[model] = float(m.group(1))
    return latencies


def load_inference_latencies(outputs_path: str) -> list:
    with open(outputs_path) as f:
        data = json.load(f)
    return [
        item["elapsed_seconds"]
        for item in data.get("items", [])
        if item.get("status") == "ok" and "elapsed_seconds" in item
    ]


query_latencies = parse_query_latencies(RESULTS_PATH)
inference_times = load_inference_latencies(LONG_OUTPUTS_PATH)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Latency Comparison", fontsize=15, fontweight="bold")

# --- Left: query latency bar chart ---
ax = axes[0]
models = [m for m in MODEL_LABELS if m in query_latencies]
values = [query_latencies[m] for m in models]
colors = [COLORS[m] for m in models]
display_names = [MODEL_DISPLAY[m] for m in models]

bars = ax.bar(display_names, values, color=colors, alpha=0.85, width=0.5)
for bar, v in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{v:.1f} ms",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

baseline = values[0]
for bar, v, name in zip(bars[1:], values[1:], display_names[1:]):
    delta = v - baseline
    sign = "+" if delta >= 0 else ""
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"({sign}{delta:.1f} ms vs Base)",
        ha="center", va="bottom", fontsize=8.5, color="#666666"
    )

ax.set_title("Avg Query Latency per Model\n(ChromaDB cosine search, 74 docs)", fontsize=11)
ax.set_ylabel("Milliseconds", fontsize=10)
ax.set_ylim(0, max(values) * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Right: inference time distribution (long model only) ---
ax2 = axes[1]
ax2.hist(inference_times, bins=15, color=COLORS["long_quantized"], alpha=0.8, edgecolor="white")
mean_t = np.mean(inference_times)
median_t = np.median(inference_times)
ax2.axvline(mean_t, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Mean: {mean_t:.1f}s")
ax2.axvline(median_t, color="#2c3e50", linestyle=":", linewidth=1.5, label=f"Median: {median_t:.1f}s")
ax2.set_title(f"Video Inference Time Distribution\n(Long model, n={len(inference_times)} videos)", fontsize=11)
ax2.set_xlabel("Seconds per video", fontsize=10)
ax2.set_ylabel("Count", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {OUTPUT_IMAGE}")

print("\nQuery latencies:")
for m in models:
    print(f"  {MODEL_DISPLAY[m]}: {query_latencies[m]:.2f} ms")
print(f"\nInference times (long model): mean={mean_t:.2f}s  median={median_t:.2f}s  min={min(inference_times):.2f}s  max={max(inference_times):.2f}s")
