from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch 
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}") # Should print 2
print(f"Current Device: {torch.cuda.current_device()}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
 
# default: Load the model on the available device(s)
# model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-30B-A3B-Thinking", dtype="auto", 
#     device_map="auto",
#     # local_files_only=True,
# )
# max_memory_mapping = {
#     0: "12GiB",  # Limit weights here to leave 12GB free for Vision/Video
#     1: "20GiB", 
#     2: "20GiB"
# }

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    quantization_config=bnb_config, # This is enough
    device_map="auto",
    # max_memory=max_memory_mapping,
    attn_implementation="sdpa",
    dtype="auto" # Ensure weights match compute-dtype
)





processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/user/home/ym22470/work/videos/zip_folder/01ff5bb49c2ff92fa41f98dbd1da45a4e9653d215161de8f58647e87af7daddd.mp4",
                # Reduce resolution and frame rate significantly
                "fps": 1, 
                "max_frames": 8, 
            },
            {"type": "text", "text": "Describe the content of this clip, in the format of tags (the main objects in the video), mood(the overall feeling or atmosphere), and action(the main activities or movements). Focus on the most salient aspects."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
