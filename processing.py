import torch
from typing import Dict, Any, List

def process_logits(logits: torch.Tensor, tokenizer: Any, top_k: int = 5) -> Dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    top1_prob = top_probs[0].item()
    top2_prob = top_probs[1].item() if top_k > 1 else 0.0
    delta_p = top1_prob - top2_prob

    candidates: List[Dict[str, Any]] = []
    for prob, token_id in zip(top_probs, top_indices):
        token_str = tokenizer.decode([token_id.item()])
        candidates.append({
            "token": token_str,
            "token_id": token_id.item(),
            "probability": prob.item(),
            "percentage": round(prob.item() * 100, 2)
        })

    return {
        "candidates": candidates,
        "top1_token": candidates[0]["token"],
        "top1_prob": top1_prob,
        "delta_p": delta_p
    }


if __name__ == "__main__":
    from model_loader import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer()

    messages = [{"role": "user", "content": "Türkiye'nin başkenti neresidir?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_token_logits = outputs.logits[0, -1, :]
    result = process_logits(last_token_logits, tokenizer)

    print(f"Top 1 Token : '{result['top1_token']}'")
    print(f"Top 1 Prob  : {result['top1_prob'] * 100:.2f}%")
    print(f"Delta P     : {result['delta_p'] * 100:.2f}%\n")

    for item in result["candidates"]:
        print(f"Token: '{item['token']:<10}' | ID: {item['token_id']:<6} | Percentage: {item['percentage']}%")