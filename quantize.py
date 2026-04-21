import torch
import gradio as gr
import numpy as np
from decord import VideoReader, cpu
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# =========================
# CONFIG
# =========================

MODEL_PATH = "/user/home/ym22470/work/models/qwen2_5_vl_quantized"

MAX_FRAMES = 16   # RTX 4070 laptop safe
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=True
)

print("Loading quantized model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model.eval()

print("Model loaded successfully!")

# =========================
# VIDEO LOADER
# =========================

def load_video_frames(video_path, max_frames=MAX_FRAMES):

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(
            0,
            total_frames - 1,
            max_frames
        ).astype(int)

    frames = vr.get_batch(indices).asnumpy()

    pil_frames = [Image.fromarray(frame) for frame in frames]

    return pil_frames

# =========================
# INFERENCE FUNCTION
# =========================

def run_inference(video, prompt):

    if video is None:
        return "Please upload a video."

    try:

        frames = load_video_frames(video, MAX_FRAMES)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            videos=[frames],
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        response = processor.batch_decode(
            output,
            skip_special_tokens=True,
        )[0]

        return response

    except Exception as e:
        return f"Error: {str(e)}"


# =========================
# GRADIO UI
# =========================

title = "Qwen2.5-VL Video Understanding Demo"

description = """
Upload a video and ask questions about it.

Optimized for local RTX 4070 laptop deployment.
"""

interface = gr.Interface(

    fn=run_inference,

    inputs=[
        gr.Video(label="Upload Video"),
        gr.Textbox(
            label="Prompt",
            value="Describe this video in detail.",
        ),
    ],

    outputs=gr.Textbox(label="Model Response"),

    title=title,
    description=description,

)

# =========================
# LAUNCH
# =========================

if __name__ == "__main__":

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )