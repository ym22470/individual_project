from datasets import load_dataset
import json

def download_sharegpt_subset(limit, output_file="sharegpt4video_subset120.json"):
    print(f"Connecting to Hugging Face...")
    
    # Load the dataset in streaming mode to save disk space and time
    # Dataset name: 'ShareGPT4Video/ShareGPT4Video'
    dataset = load_dataset("ShareGPT4Video/ShareGPT4Video", split="train", streaming=True)
    
    subset_data = []
    
    print(f"Fetching the first {limit} entries...")
    
    # Iterate through the stream and collect data
    for i, entry in enumerate(dataset):
        if len(subset_data) >= limit:
            break

        if entry.get("zip_folder") == "pixabay_videos_1.zip":
            subset_data.append(entry)
        
            if len(subset_data) % 20 == 0:
                print(f"Progress: {len(subset_data)}/{limit}")

    # Save to a local JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(subset_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nDone! Saved {len(subset_data)} entries to {output_file}")

if __name__ == "__main__":
    download_sharegpt_subset(limit=120)