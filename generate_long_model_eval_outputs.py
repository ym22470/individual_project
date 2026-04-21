import argparse
import json
import os
import time
from typing import Dict, List

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

os.environ["HF_TOKEN"] = "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"

DEFAULT_LONG_MODEL_PATH = "/home/chongshengwang/naratix/qwen2_5_vl_long_quantized"
DEFAULT_VIDEOS_DIR = "/home/chongshengwang/naratix/eval_videos"
DEFAULT_OUTPUT_PATH = "/home/chongshengwang/naratix/benchmark_cache/eval_videos_long_outputs.json"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def list_video_files(videos_dir: str) -> List[str]:
    return [
        os.path.join(videos_dir, file_name)
        for file_name in sorted(os.listdir(videos_dir))
        if file_name.lower().endswith(VIDEO_EXTENSIONS)
    ]


def describe_video(model, processor, video_path: str, fps: float, max_new_tokens: int) -> str:
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

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]


def load_existing_output(output_path: str) -> Dict:
    if not os.path.exists(output_path):
        return {"items": []}

    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_output(output_path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = f"{output_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the long Qwen2.5-VL model on all eval_videos and save outputs to JSON"
    )
    parser.add_argument("--videos-dir", default=DEFAULT_VIDEOS_DIR, help="Directory containing evaluation videos")
    parser.add_argument("--model-path", default=DEFAULT_LONG_MODEL_PATH, help="Path to the long quantized model")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_PATH, help="Path to the JSON output file")
    parser.add_argument("--fps", type=float, default=0.5, help="Sampling FPS for video analysis")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of generated tokens")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing JSON output and regenerate all entries",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.videos_dir):
        raise RuntimeError(f"Videos directory not found: {args.videos_dir}")
    if not os.path.exists(args.model_path):
        raise RuntimeError(f"Model path not found: {args.model_path}")

    video_files = list_video_files(args.videos_dir)
    if not video_files:
        raise RuntimeError(f"No video files found in {args.videos_dir}")

    existing_payload = {"items": []} if args.overwrite else load_existing_output(args.output_file)
    existing_items = existing_payload.get("items", [])
    items_by_path = {item["video_path"]: item for item in existing_items if "video_path" in item}

    print(f"Found {len(video_files)} videos in {args.videos_dir}")
    print(f"Loading model from: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    payload = {
        "model_label": "long_quantized",
        "model_path": args.model_path,
        "videos_dir": args.videos_dir,
        "fps": args.fps,
        "max_new_tokens": args.max_new_tokens,
        "video_count": len(video_files),
        "items": [],
    }

    for index, video_path in enumerate(video_files, start=1):
        if video_path in items_by_path:
            existing_item = items_by_path[video_path]
            payload["items"].append(existing_item)
            print(f"[{index}/{len(video_files)}] Skipping existing result for {existing_item['file_name']}")
            continue

        file_name = os.path.basename(video_path)
        print(f"[{index}/{len(video_files)}] Processing {file_name}")
        started_at = time.perf_counter()

        try:
            description = describe_video(
                model=model,
                processor=processor,
                video_path=video_path,
                fps=args.fps,
                max_new_tokens=args.max_new_tokens,
            )
            item = {
                "file_name": file_name,
                "video_path": video_path,
                "description": description,
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "status": "ok",
            }
            print(f"  Output: {description[:160]}...")
        except Exception as exc:
            item = {
                "file_name": file_name,
                "video_path": video_path,
                "description": "",
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "status": "error",
                "error": str(exc),
            }
            print(f"  Error: {exc}")

        payload["items"].append(item)
        save_output(args.output_file, payload)

    ok_count = sum(1 for item in payload["items"] if item.get("status") == "ok")
    error_count = sum(1 for item in payload["items"] if item.get("status") == "error")
    print(f"\nSaved results to {args.output_file}")
    print(f"Successful videos: {ok_count}")
    print(f"Errored videos: {error_count}")


if __name__ == "__main__":
    main()