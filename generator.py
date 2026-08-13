import torch
from typing import Dict, Any, Generator
from model_loader import load_model_and_tokenizer
from processing import process_logits
from metrics import calculate_entropy, EntropyTracker

def generate_stream(prompt: str, 
                    max_new_tokens: int =20,
                    model: Any = None,
                    tokenizer: Any = None)-> Generator[Dict[str,Any], None, None]:
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()
    tracker = EntropyTracker(window_size=20)

    # format input
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids= inputs["input_ids"]
    eos_token_id=tokenizer.eos_token_id

    step = 0
    while step < max_new_tokens:
        with torch.no_grad():
            outputs = model (input_ids)

        # logits for the last generated position
        last_token_logits = outputs.logits[0, -1, :]

        processed = process_logits(last_token_logits, tokenizer)

        probabilities = torch.softmax(last_token_logits, dim= -1)
        entropy = calculate_entropy(probabilities)
        stats= tracker.update(entropy)

        top1_id = torch.argmax(last_token_logits, dim=-1).item()
        if top1_id == eos_token_id:
            break

        next_token_tensor = torch.tensor([[top1_id]])
        input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)

        yield{
            "token": processed["top1_token"],
            "token_id": top1_id,
            "candidates": processed["candidates"],
            "delta_p": processed["delta_p"],
            "entropy": entropy,
            "z_score": stats["z_score"],
            "mean_entropy": stats["mean"]
        }        

        step +=1

if __name__ == "__main__":
    print("***** Streaming Generation Test *****")
    for packet in generate_stream("Türkiye'nin başkenti neresidir?", max_new_tokens=10):
        print(f"Token: '{packet['token']:<10}' | H: {packet['entropy']:.2f} | Z: {packet['z_score']:.2f} | Delta P: {packet['delta_p']*100:.1f}%")        