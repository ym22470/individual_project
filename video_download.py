import json
from huggingface_hub import hf_hub_download
import os

# Create a folder for the videos
os.makedirs("videos", exist_ok=True)

# Path to the metadata we just downloaded
metadata_path = "sharegpt4video_40k.jsonl"
limit = 5

print(f"Starting download for first {limit} videos...")

with open(metadata_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= limit:
            break
            
        data = json.loads(line)
        video_path = data['video_path'] # Example: "pexels/12345.mp4"
        
        # ShareGPT4Video stores videos inside a 'zip_folder' directory on HF
        # The 'video' key usually maps to the path inside that folder
        hf_path = f"zip_folder/{video_path}"
        
        try:
            print(f"[{i+1}/{limit}] Downloading: {video_path}")
            hf_hub_download(
                repo_id="ShareGPT4Video/ShareGPT4Video",
                filename=hf_path,
                repo_type="dataset",
                local_dir="videos",
                local_dir_use_symlinks=False # Best for Linux to avoid link issues
            )
        except Exception as e:
            print(f"Error downloading {video_path}: {e}")

print("\nDownload complete. Check the 'videos' folder.")