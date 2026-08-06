import os
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def load_model_and_tokenizer():
    """
    Day 2: Loads model and tokenizer into RAM, measures RAM/VRAM usage.
    """
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model into RAM: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map="cpu"
    )
    
    # RAM and VRAM Check
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    
    print("\n--- Memory Inspection ---")
    print(f"RAM Usage: {ram_gb:.2f} GB")
    print("VRAM Usage: 0.00 GB (CPU Mode)")
    print("-------------------------\n")
    
    return model, tokenizer

if __name__ == "__main__":
    load_model_and_tokenizer()