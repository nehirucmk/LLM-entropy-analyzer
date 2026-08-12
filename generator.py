import torch
from typing import List
from model_loader import load_model_and_tokenizer

def generate_step_by_step(prompt: str, max_new_tokens: int = 20):
    model, tokenizer =load_model_and_tokenizer()

    # format input with chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt= True
    )

    # formatted prompt -> tensor
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    eos_token_id = tokenizer.eos_token_id
    generated_tokens: List[int] = []

    print(f"Prompt: {prompt}\nGenerating: ", end="", flush=True)

    # custom generation loop
    step = 0
    while step < max_new_tokens:
        with torch.no_grad():
            outputs = model(input_ids)

        next_token_logits = outputs.logits[0, -1, :]

        next_token_id = torch.argmax(next_token_logits, dim=-1).item()

        if next_token_id == eos_token_id:
            break

        generated_tokens.append(next_token_id)
        next_token_tensor = torch.tensor([[next_token_id]])
        input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)

        token_str = tokenizer.decode([next_token_id])
        print(token_str, end="", flush=True)

        step += 1

    print("\n\nDone.")

if __name__ == "__main__":
    generate_step_by_step("Türkiye'nin başkenti neresidir?", max_new_tokens=15)