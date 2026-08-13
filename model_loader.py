import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def load_model_and_tokenizer():
    # load tokenizer and model directly into ram without terminal clutter
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map="cpu"
    )
    return model, tokenizer

if __name__ == "__main__":
    load_model_and_tokenizer()