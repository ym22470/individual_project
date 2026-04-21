from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer
import os
os.environ["HF_TOKEN"] = "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"
# from sentence_transformers.util import cosine_similarity

# Load your local quantized model
model_path = "/home/chongshengwang/naratix/qwen2_5_vl_base_quantized"  # CHANGE to your quantized model path
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, device_map="auto", trust_remote_code=True, local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path, use_fast = True, trust_remote_code=True, local_files_only=True)

# Prepare the video input
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/home/chongshengwang/naratix/videos/6502898_Cute Girl Doing Homework_By_ChamanExperience_Artlist_HD.mp4",
                "fps": 1,  # Samples 1 frame every 1.25 seconds
            },
            {"type": "text", "text": "Analyze this video for indexing."},
        ],
    }
]

# Process and Generate
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
    skip_special_tokens=True
)
print(output_text[0])

embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
video_embedding = embedding_model.encode(output_text[0])

user_embedding = embedding_model.encode("girl writting")  # Should be similar to the video embedding
fake_input = embedding_model.encode("In a video scene, a car is driving on a highway with trees in the background.")  # Should be different from the video embedding

similarity = embedding_model.similarity(video_embedding, user_embedding)
similarity_fake = embedding_model.similarity(video_embedding, fake_input)  # Should be close to 1.0
# print("Video embedding:", video_embedding)
# print("User embedding:", user_embedding)
print("Similarity:", similarity)
print("Similarity with fake input:", similarity_fake)W