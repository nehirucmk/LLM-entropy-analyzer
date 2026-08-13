import sys
import os
import matplotlib.pyplot as plt

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_stream
from model_loader import load_model_and_tokenizer

def run_demo():
    print("Loading model into RAM...")
    model, tokenizer = load_model_and_tokenizer()
    print("Model ready!")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))

    while True:
        user_prompt = input("\nAsk me anything (type exit to quit): ")
        if user_prompt.lower() == 'exit':
            break
        if not user_prompt.strip():
            continue

        x_data, z_data, h_data = [], [], []

        print("\nGenerated Text: ", end="", flush=True)

        for i, packet in enumerate(generate_stream(user_prompt, max_new_tokens=100, model=model, tokenizer=tokenizer)):
            token = packet["token"]
            z_score = packet["z_score"]
            entropy = packet["entropy"]

            print(token, end="", flush=True)

            x_data.append(i)
            z_data.append(z_score)
            h_data.append(entropy)

            ax.clear()
            ax.plot(x_data, z_data, 'r-o', label='Z-Score', linewidth=2)
            ax.plot(x_data, h_data, 'b--s', label='Entropy (H)', linewidth=1.5)
            ax.axhline(y=2.0, color='g', linestyle=':', label='Threshold (Z=2.0)')
            ax.set_title("Real-Time LLM Telemetry")
            ax.set_xlabel("Token Step")
            ax.set_ylabel("Value")
            ax.legend(loc="upper left")
            ax.grid(True)

            plt.draw()
            plt.pause(0.001)

        print("\n\nDone.")

if __name__ == "__main__":
    run_demo()