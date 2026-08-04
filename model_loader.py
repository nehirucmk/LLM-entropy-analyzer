import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def load_model_and_tokenizer():
    """
    Downloads and loads the Qwen2.5-1.5B model and tokenizer into CPU RAM.
    """
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model {MODEL_NAME} into system RAM (CPU mode)...")

    # using float32 for maximum CPU stability and accuracy
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    print("Model and tokenizer successfully loaded.")
    return model, tokenizer

def calculate_shannon_entropy(logits):
    """
    Calculates Shannon Entropy from raw model logits.
    Formula: H(X) = - sum(P(x) * log2(P(x)))
    """
    # convert logits to probabilities using Softmax
    probabilities = torch.softmax(logits, dim=-1)

    entropy = 0.0
    for prob in probabilities[0]:
        p = prob.item()
        if p > 0:
            entropy -= p * math.log2(p)
            
    return entropy

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    
    test_prompt = "explain what an algorithm is in one simple sentence."
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    print(f"\nTest Prompt: {test_prompt}")
    
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]
        
        entropy_score = calculate_shannon_entropy(next_token_logits)
        
        print(f"Calculated Shannon Entropy for next token: {entropy_score:.4f} bits")