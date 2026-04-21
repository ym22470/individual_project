import gradio as gr
import chromadb
import os
import shutil
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration
from sentence_transformers import SentenceTransformer

# --- 1. SETTINGS & PATHS ---
# HF_TOKEN should be set in your environment: export HF_TOKEN=<your_token>
model_path = "/home/chongshengwang/naratix/qwen2_5_vl_quantized"
storage_dir = "/home/chongshengwang/naratix/permanent_videos"
os.makedirs(storage_dir, exist_ok=True)

# --- 2. MODEL LOADING ---
print("--- Loading Qwen2.5-VL model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, device_map="auto", trust_remote_code=True, local_files_only=True
)
print("--- Model loaded. Loading Processor...")
processor = AutoProcessor.from_pretrained(model_path, use_fast=True, trust_remote_code=True, local_files_only=True)

# --- 3. VECTOR DATABASE SETUP ---
class NomicEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    def __call__(self, input):
        # Automatically apply Nomic prefix for indexing
        prefixed_input = [f"search_document: {text}" for text in input]
        return self.model.encode(prefixed_input).tolist()


def embed_search_query(text):
    """Embeds a user query with the correct Nomic query prefix."""
    return embedding_fn.model.encode([f"search_query: {text}"]).tolist()[0]

chroma_client = chromadb.PersistentClient(path="video_search_db")
embedding_fn = NomicEmbeddingFunction()

print("--- Connecting to ChromaDB (HNSW + Cosine Similarity)...")
# Delete existing 'video_search_db' folder if you need to reset the distance metric
collection = chroma_client.get_or_create_collection(
    name="video_embeddings", 
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 200}
)

print("--- Cleaning up VectorDB for a fresh start...")
# This deletes everything in the collection
try:
    existing_ids = collection.get()['ids']
    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"--- Successfully cleared {len(existing_ids)} entries.")
except Exception as e:
    print(f"--- Note: Database was already empty or error occurred: {e}")

if os.path.exists(storage_dir):
    shutil.rmtree(storage_dir)
    os.makedirs(storage_dir)

# --- 4. CORE FUNCTIONS ---

def index_video(temp_video_path):
    """Processes a single video: Move to storage -> Qwen-VL -> ChromaDB Index."""
    try:
        # Move video from Gradio's /tmp to permanent storage
        file_name = os.path.basename(temp_video_path)
        permanent_path = os.path.join(storage_dir, file_name)
        exsisting = os.path.exists(permanent_path)
        if exsisting:
            return f"File {file_name} already indexed."

        shutil.copy(temp_video_path, permanent_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": permanent_path, "fps": 0.5},
                {"type": "text", "text": "Analyze this video for indexing."},
            ],
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, 
            padding=True, return_tensors="pt"
        ).to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=256)
        output_text = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]

        # Use the permanent path as ID so search results find the actual file
        video_embedding = embedding_fn([output_text])[0]
        collection.add(
            ids=[permanent_path],
            embeddings=[video_embedding],
            metadatas=[{"description": output_text}],
            documents=[output_text]
        )
        return f"✅ Indexed: {file_name}\nModel: {output_text[:50]}..."
    except Exception as e:
        return f"❌ Error indexing {temp_video_path}: {str(e)}"

def index_multiple_videos(video_files):
    """Wrapper to handle multiple file uploads from Gradio."""
    if not video_files:
        return "No files detected."
    results = []
    for vf in video_files:
        res = index_video(vf.name)
        results.append(res)
    return "\n".join(results)

def search_and_display(search_query, threshold=0.55):
    """Searches the HNSW index and returns (path, label) for the Gallery."""
    if not search_query:
        return []

    query_embedding = embed_search_query(search_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=6,
        include=["metadatas", "distances"]
    )
    
    gallery_items = []
    if results['ids'][0]:
        for i in range(len(results['ids'][0])):
            video_path = results['ids'][0][i]
            # normalized cosine score: 1 - distance
            score = 1 - results['distances'][0][i]
            if score >= threshold:
                desc = results['metadatas'][0][i]['description']
                label = f"Match {i+1} (Score: {score:.2f})\n{desc}"
                gallery_items.append((video_path, label))
    
    return gallery_items

def clear_search_results():
    """Clears search input and all currently displayed videos in the gallery."""
    return "", []

# --- 5. GRADIO UI LAYOUT ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Narratix Video Indexing & Search Demo")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Batch Indexing")
            file_input = gr.File(label="Upload Videos", file_count="multiple", file_types=["video"])
            index_button = gr.Button("Analyze & Index", variant="primary")
            index_status = gr.Textbox(label="Processing Status", lines=10)
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. Semantic Search")
            search_input = gr.Textbox(placeholder="Try: 'A deer in a forest' or 'fast car tunnel'", label="Query")
            with gr.Row():
                search_button = gr.Button("Search Library", variant="secondary")
                clear_button = gr.Button("Clear Output Videos")
            # The Gallery can take a list of (video_path, label) tuples
            video_gallery = gr.Gallery(
                label="Search Results", 
                columns=3, 
                height="auto", 
                object_fit="contain",
                allow_preview=True
            )

    # Wire up the buttons
    index_button.click(index_multiple_videos, inputs=[file_input], outputs=[index_status])
    search_button.click(search_and_display, inputs=[search_input], outputs=[video_gallery])
    clear_button.click(clear_search_results, outputs=[search_input, video_gallery])

# --- 6. LAUNCH ---
if __name__ == "__main__":
    demo.launch(
        server_name="192.168.112.194", 
        server_port=7860,
        allowed_paths=["/home/chongshengwang/naratix/"] # Crucial for Gradio to access your local videos
    )