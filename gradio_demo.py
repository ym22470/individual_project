from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load your local quantized model
model_path = "/user/home/ym22470/work/models/qwen2_5_vl_quantized"  # CHANGE to your quantized model path
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, device_map="auto", trust_remote_code=True, local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path)

# Prepare the video input
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/user/home/ym22470/work/12332642_1080_1918_30fps.mp4",
                "fps": 1,  # Samples 1 frame every 1 second
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
