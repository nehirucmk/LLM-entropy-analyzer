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
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15
) -> Generator[Dict[str, Any], None, None]:
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    tracker = EntropyTracker(window_size=20)

    # format prompt with chat template
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

    # collect valid stop tokens for qwen architecture
    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    for extra_stop in ["<|im_end|>", "<|endoftext|>"]:
        token_id = tokenizer.convert_tokens_to_ids(extra_stop)
        if token_id is not None and isinstance(token_id, int):
            stop_token_ids.add(token_id)

    step = 0
    while step < max_new_tokens:
        with torch.no_grad():
            outputs = model(input_ids)

        last_token_logits = outputs.logits[0, -1, :].clone()

        # extract telemetric measurements from unmodified logits
        raw_probs = torch.softmax(last_token_logits, dim=-1)
        entropy = calculate_entropy(raw_probs)
        stats = tracker.update(entropy)
        processed = process_logits(last_token_logits, tokenizer)

        # apply repetition penalty to penalize previously seen tokens
        if repetition_penalty != 1.0:
            for prev_token_id in set(input_ids[0].tolist()):
                if last_token_logits[prev_token_id] < 0:
                    last_token_logits[prev_token_id] *= repetition_penalty
                else:
                    last_token_logits[prev_token_id] /= repetition_penalty

        # temperature and top-p sampling
        if temperature > 0:
            scaled_logits = last_token_logits / max(temperature, 1e-5)
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scaled_logits[indices_to_remove] = float('-inf')

            filtered_probs = torch.softmax(scaled_logits, dim=-1)
            selected_token_id = torch.multinomial(filtered_probs, num_samples=1).item()
        else:
            selected_token_id = torch.argmax(last_token_logits, dim=-1).item()

        # stop generation if stop token is emitted
        if selected_token_id in stop_token_ids:
            break

        # append selected token to sequence context
        next_token_tensor = torch.tensor([[selected_token_id]])
        input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)

        # decode actual selected token string
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