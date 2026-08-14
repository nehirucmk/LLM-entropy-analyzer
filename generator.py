import sys
import torch
from typing import Dict, Any, Generator
from model_loader import load_model_and_tokenizer
from processing import process_logits
from metrics import calculate_entropy, EntropyTracker

def generate_stream(
    prompt: str, 
    max_new_tokens: int = 80, 
    model: Any = None, 
    tokenizer: Any = None,
    temperature: float = 0.0,
    top_k: int = 50,
    window_size: int = 20
) -> Generator[Dict[str, Any], None, None]:
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    tracker = EntropyTracker(window_size=window_size)

    # prepare prompt with chat template
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

    # collect stop tokens
    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    for extra_stop in ["<|im_end|>", "<|endoftext|>"]:
        token_id = tokenizer.convert_tokens_to_ids(extra_stop)
        if token_id is not None and isinstance(token_id, int):
            stop_token_ids.add(token_id)

    step = 0
    past_key_values = None
    current_input_ids = input_ids

    while step < max_new_tokens:
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(current_input_ids, use_cache=True)
            else:
                outputs = model(current_input_ids, past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs.past_key_values

        # telemetry calculations on clean logits
        raw_logits = outputs.logits[0, -1, :]
        raw_probs = torch.softmax(raw_logits, dim=-1)
        entropy = calculate_entropy(raw_probs)
        stats = tracker.update(entropy)
        processed = process_logits(raw_logits, tokenizer, top_k=5)

        # token selection
        if temperature <= 1e-4:
            selected_token_id = torch.argmax(raw_logits, dim=-1).item()
        else:
            scaled_logits = raw_logits / temperature
            top_values, top_indices = torch.topk(scaled_logits, k=top_k)
            safe_probs = torch.softmax(top_values, dim=-1)
            choice_idx = torch.multinomial(safe_probs, num_samples=1).item()
            selected_token_id = top_indices[choice_idx].item()

        # stop on end token
        if selected_token_id in stop_token_ids:
            break

        # advance sequence with only 1 token (O(1) memory pass)
        current_input_ids = torch.tensor([[selected_token_id]])
        selected_token_str = tokenizer.decode([selected_token_id])

        yield {
            "token": selected_token_str,
            "token_id": selected_token_id,
            "candidates": processed["candidates"],
            "delta_p": processed["delta_p"],
            "entropy": entropy,
            "z_score": stats["z_score"],
            "mean_entropy": stats["mean"]
        }

        step += 1


if __name__ == "__main__":
    try:
        model, tokenizer = load_model_and_tokenizer()

        test_prompt = "What is the chemical symbol for gold?"
        print(f"\nPrompt: {test_prompt}\nGenerating: ", end="", flush=True)

        for packet in generate_stream(test_prompt, max_new_tokens=40, model=model, tokenizer=tokenizer):
            print(packet["token"], end="", flush=True)

        print("\n\n[Done]")
    except KeyboardInterrupt:
        print("\n\n[Process terminated by user]")
        sys.exit(0)