import argparse
import json
import os
import re
import time
from typing import Dict, List, Set

from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

os.environ["HF_TOKEN"] = "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"

DEFAULT_BASE_MODEL_PATH = "/home/chongshengwang/naratix/qwen2_5_vl_base_quantized"
DEFAULT_LONG_MODEL_PATH = "/home/chongshengwang/naratix/qwen2_5_vl_long_quantized"
DEFAULT_TAGS_MODEL_PATH = "/home/chongshengwang/naratix/qwen2_5_vl_tags_quantized"
DEFAULT_QUERIES_PATH = "/home/chongshengwang/naratix/benchmark_queries.jsonl"


def to_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def canonicalize_video_key(value: str) -> str:
    stem = os.path.splitext(os.path.basename(value or ""))[0].lower()
    stem = re.sub(r"[^\w]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"(?:^|_)(?:artlist|artist)(?=_|$)", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")


def load_query_rows(jsonl_path: str) -> List[Dict[str, str]]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            row = json.loads(stripped)
            query = (row.get("query") or "").strip()
            target_value = (
                row.get("target_id")
                or row.get("video_id")
                or row.get("file_name")
                or row.get("video")
                or row.get("path")
            )

            if not query:
                raise ValueError(f"Missing query in {jsonl_path} at line {line_number}")
            if not target_value:
                raise ValueError(
                    f"Missing target_id/video_id/file_name/video/path in {jsonl_path} at line {line_number}"
                )

            rows.append({"query": query, "target_value": str(target_value).strip()})

    if not rows:
        raise RuntimeError(f"No benchmark queries found in {jsonl_path}")

    return rows


def resolve_benchmark_queries(
    query_rows: List[Dict[str, str]],
    video_entries: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    key_to_doc_id = {}

    for entry in video_entries:
        for candidate in (entry["id"], entry["file_name"], entry["path"]):
            key = canonicalize_video_key(candidate)
            key_to_doc_id[key] = entry["id"]

    benchmark = []
    for row in query_rows:
        target_key = canonicalize_video_key(row["target_value"])
        doc_id = key_to_doc_id.get(target_key)
        if doc_id is None:
            continue

        benchmark.append({"query": row["query"], "target_id": doc_id})

    if not benchmark:
        raise RuntimeError("No benchmark queries in the JSONL file matched the selected videos.")

    return benchmark


def describe_video(model, processor, video_path: str, fps: float = 0.5) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "text", "text": "Analyze this video for indexing."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    output_text = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]
    return output_text


def cosine_score(embedding_model, query_text: str, doc_embedding) -> float:
    query_embedding = embedding_model.encode(f"search_query: {query_text}")
    return to_float(embedding_model.similarity(doc_embedding, query_embedding))


def evaluate(
    embedding_model,
    docs: List[Dict[str, str]],
    doc_embeddings: Dict[str, List[float]],
    benchmark_queries: List[Dict[str, str]],
):
    cosine_hits = 0
    cosine_hits_at_5 = 0
    cosine_mrr = 0.0
    cosine_latency = 0.0

    print("\nPer-query top results:")
    print("-" * 80)

    for row in benchmark_queries:
        query = row["query"]
        target_id = row["target_id"]

        t0 = time.perf_counter()
        cosine_ranked = []
        for doc in docs:
            score = cosine_score(embedding_model, query, doc_embeddings[doc["id"]])
            cosine_ranked.append((score, doc["id"]))
        cosine_ranked.sort(key=lambda x: x[0], reverse=True)
        cosine_latency += time.perf_counter() - t0

        cosine_top1 = cosine_ranked[0][1]
        cosine_top5 = [doc_id for _, doc_id in cosine_ranked[:5]]

        if cosine_top1 == target_id:
            cosine_hits += 1
        if target_id in cosine_top5:
            cosine_hits_at_5 += 1

        for rank, (_, doc_id) in enumerate(cosine_ranked, start=1):
            if doc_id == target_id:
                cosine_mrr += 1.0 / rank
                break

        print(f"Query: {query}")
        print(f"  Target: {target_id}")
        print(f"  Cosine top-1: {cosine_top1}")
        print("-" * 80)

    n = max(len(benchmark_queries), 1)
    return {
        "cosine_recall_at_1": cosine_hits / n,
        "cosine_recall_at_5": cosine_hits_at_5 / n,
        "cosine_mrr": cosine_mrr / n,
        "cosine_avg_latency_ms": (cosine_latency / n) * 1000,
    }


def run_model_benchmark(
    model_path: str,
    model_label: str,
    selected_videos: List[str],
    query_rows: List[Dict[str, str]],
    embedding_model,
    fps: float,
):
    print(f"\n=== Running benchmark for {model_label} ===")
    print(f"Loading model from: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    docs = []
    for idx, video_path in enumerate(selected_videos, start=1):
        file_name = os.path.basename(video_path)
        print(f"[{model_label}] [{idx}/{len(selected_videos)}] Analyzing {file_name}...")
        description = describe_video(model, processor, video_path, fps=fps)
        docs.append(
            {
                "id": f"video_{idx}",
                "file_name": file_name,
                "path": video_path,
                "text": description,
            }
        )
        print(f"  Description: {description[:120]}...")

    doc_embeddings = {
        doc["id"]: embedding_model.encode(f"search_document: {doc['text']}")
        for doc in docs
    }

    benchmark_queries = resolve_benchmark_queries(query_rows, docs)

    print(f"\nBenchmark set for {model_label}:")
    for row in benchmark_queries:
        print(f"  - query='{row['query']}' -> target={row['target_id']}")

    metrics = evaluate(
        embedding_model=embedding_model,
        docs=docs,
        doc_embeddings=doc_embeddings,
        benchmark_queries=benchmark_queries,
    )

    print(f"\n=== Aggregate Metrics ({model_label}) ===")
    print(f"Cosine Recall@1: {metrics['cosine_recall_at_1']:.3f}")
    print(f"Cosine Recall@5: {metrics['cosine_recall_at_5']:.3f}")
    print(f"Cosine MRR:      {metrics['cosine_mrr']:.3f}")
    print(f"Cosine avg latency/query: {metrics['cosine_avg_latency_ms']:.2f} ms")

    return {
        "label": model_label,
        "metrics": metrics,
        "query_count": len(benchmark_queries),
    }


def print_model_comparison(results: List[Dict]):
    print("\n=== Model Comparison ===")
    print(f"Queries evaluated: {results[0]['query_count']}")

    label_width = 30
    value_width = 12
    headers = [result["label"] for result in results]
    header_row = f"{'Metric':<{label_width}}" + "".join(f"{header:>{value_width}}" for header in headers)
    print(header_row)
    print("-" * len(header_row))

    metric_rows = [
        ("Cosine Recall@1", "cosine_recall_at_1"),
        ("Cosine Recall@5", "cosine_recall_at_5"),
        ("Cosine MRR", "cosine_mrr"),
    ]

    for title, key in metric_rows:
        row = f"{title:<{label_width}}" + "".join(
            f"{result['metrics'][key]:>{value_width}.3f}" for result in results
        )
        print(row)

    latency_rows = [
        ("Cosine latency (ms)", "cosine_avg_latency_ms"),
    ]
    for title, key in latency_rows:
        row = f"{title:<{label_width}}" + "".join(
            f"{result['metrics'][key]:>{value_width}.2f}" for result in results
        )
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Batch test cosine-only vs hybrid retrieval on local videos")
    parser.add_argument(
        "--videos-dir",
        default="/home/chongshengwang/naratix/eval_videos",
        help="Directory containing test videos",
    )
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_PATH, help="Path to base quantized model")
    parser.add_argument("--long-model-path", default=DEFAULT_LONG_MODEL_PATH, help="Path to long quantized model")
    parser.add_argument("--tags-model-path", default=DEFAULT_TAGS_MODEL_PATH, help="Path to tags quantized model")
    parser.add_argument("--skip-long", action="store_true", help="Skip running long model benchmark")
    parser.add_argument("--skip-tags", action="store_true", help="Skip running tags model benchmark")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Number of videos to test. Defaults to all videos in the directory.",
    )
    parser.add_argument("--fps", type=float, default=0.5, help="Sampling FPS for video analysis")
    parser.add_argument(
        "--queries-file",
        default=DEFAULT_QUERIES_PATH,
        help="JSONL file containing benchmark queries and target video identifiers",
    )
    args = parser.parse_args()

    video_files = [
        os.path.join(args.videos_dir, f)
        for f in sorted(os.listdir(args.videos_dir))
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm"))
    ]

    if not video_files:
        raise RuntimeError(f"No video files found in {args.videos_dir}")

    if not os.path.exists(args.queries_file):
        raise RuntimeError(f"Queries file not found: {args.queries_file}")

    selected_videos = video_files if args.max_videos is None else video_files[: args.max_videos]
    print(f"Selected {len(selected_videos)} videos for testing.")
    print(f"Loading queries from: {args.queries_file}")
    query_rows = load_query_rows(args.queries_file)

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    model_runs = [
        ("base_quantized", args.base_model_path, False),
        ("long_quantized", args.long_model_path, args.skip_long),
        ("tags_quantized", args.tags_model_path, args.skip_tags),
    ]

    results = []
    for model_label, model_path, should_skip in model_runs:
        if should_skip:
            print(f"\nSkipped {model_label} benchmark due to skip flag.")
            continue
        if not os.path.exists(model_path):
            print(f"\nModel path not found for {model_label}: {model_path}")
            print(f"Skipping {model_label} benchmark.")
            continue

        results.append(
            run_model_benchmark(
                model_path=model_path,
                model_label=model_label,
                selected_videos=selected_videos,
                query_rows=query_rows,
                embedding_model=embedding_model,
                fps=args.fps,
            )
        )

    if not results:
        raise RuntimeError("No model benchmarks were run.")

    if len(results) > 1:
        print_model_comparison(results)


if __name__ == "__main__":
    main()
