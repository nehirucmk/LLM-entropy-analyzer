import torch
from typing import Dict, Any, Generator, List
from model_loader import load_model_and_tokenizer
from processing import process_logits
from metrics import calculate_entropy, EntropyTracker

def generate_stream(
    prompt: str, 
    max_new_tokens: int = 80, 
    model: Any = None, 
    tokenizer: Any = None,
    temperature: float = 0.3,
    top_p: float = 0.9
) -> Generator[Dict[str, Any], None, None]:
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    tracker = EntropyTracker(window_size=20)

    messages = [
        {"role": "system", "content": "You are a concise, accurate AI assistant. Answer directly and briefly."},
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    stop_token_ids = {tokenizer.eos_token_id}
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None:
        stop_token_ids.add(im_end_id)

    step = 0
    while step < max_new_tokens:
        with torch.no_grad():
            outputs = model(input_ids)

        last_token_logits = outputs.logits[0, -1, :]

        raw_probs = torch.softmax(last_token_logits, dim=-1)
        entropy = calculate_entropy(raw_probs)
        stats = tracker.update(entropy)
        processed = process_logits(last_token_logits, tokenizer)

        scaled_logits = last_token_logits / max(temperature, 1e-5)
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        scaled_logits[indices_to_remove] = float('-inf')

        filtered_probs = torch.softmax(scaled_logits, dim=-1)
        top1_id = torch.multinomial(filtered_probs, num_samples=1).item()

        if top1_id in stop_token_ids:
            break

        next_token_tensor = torch.tensor([[top1_id]])
        input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)

        yield {
            "token": processed["top1_token"],
            "token_id": top1_id,
            "candidates": processed["candidates"],
            "delta_p": processed["delta_p"],
            "entropy": entropy,
            "z_score": stats["z_score"],
            "mean_entropy": stats["mean"]
        }

        step += 1