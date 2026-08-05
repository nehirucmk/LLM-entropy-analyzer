import torch
from model_loader import load_model_and_tokenizer

def analyze_logits(prompt: str):
    # load model and tokenizer 
    model, tokenizer = load_model_and_tokenizer()

    # convert prompt text into token ids
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print("***** Tokenization *****")
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids[0]]}")
    print(f"Token IDs: {input_ids[0].tolist()}\n")

    with torch.no_grad():
        outputs = model(**inputs)

    # get logits for the last token position
    last_token_logits = outputs.logits[0, -1, :]

    print("***** Logit Inspection *****")
    print(f"Vocabulary Size: {last_token_logits.shape[0]}")
    print(f"Max Logit Score: {last_token_logits.max().item():.2f}")
    print(f"Min Logit Score: {last_token_logits.min().item():.2f}\n")

    # convert raw logits to probabilities
    probabilities = torch.softmax(last_token_logits, dim=-1)

    # get top 5 token candidates
    top5_probs, top5_indices = torch.topk(probabilities, k=5)

    print("***** Top 5 Candidates *****")
    for idx, (prob, token_id) in enumerate(zip(top5_probs, top5_indices), start=1):
        token_str = tokenizer.decode([token_id.item()])
        logit_val = last_token_logits[token_id].item()
        print(f"{idx}. Token: '{token_str}' | ID: {token_id.item()} | Logit: {logit_val:.2f} | Probability: {prob.item() * 100:.2f}%")

    # calculate difference between top 1 and top 2 choices
    delta_p = (top5_probs[0] - top5_probs[1]).item()
    print(f"\nDelta P (P1 - P2): {delta_p:.4f} ({delta_p * 100:.2f}%)")

if __name__ == "__main__":
    test_prompt = "Türkiye'nin başkenti"
    analyze_logits(test_prompt) 
  