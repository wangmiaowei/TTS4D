"""Download the released TTS4D weights from the Hugging Face Hub into ./weights/."""
import os
from huggingface_hub import hf_hub_download

os.makedirs("weights", exist_ok=True)
path = hf_hub_download("miaoweiwang/TTS4D", "tts4d.safetensors", local_dir="weights")
print("weights:", path)
