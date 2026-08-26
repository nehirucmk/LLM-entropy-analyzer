import os 
import sys
import gc 
import psutil
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_loader import load_model_and_tokenizer
from generator import generate_stream

def get_process_memory_mb()-> float:
    process =psutil.Process(os.getpid())
    return process.memory_info().rss/(1024*1024)

def run_test_leak(iterations:int=20):
    print("[*]loading model and tokenizer into memory")
    start_mem = get_process_memory_mb()
    model,tokenizer =load_model_and_tokenizer()
    baseline_mem = get_process_memory_mb()

    print(f"[+] baseline model memory: {baseline_mem:.2f} mb (+{baseline_mem - start_mem:.2f} mb delta)")
    prompt = "Explain quantum computing principles in three concise sentences."
    checkpoints = []

    for i in range(1, iterations+1):
        stream = generate_stream(
            prompt=prompt,
            max_new_tokens=40,
            model=model,
            tokenizer=tokenizer,
            temperature=0.0,
            top_k=50,
            window_size=20
        )
        for _ in stream: pass

        gc.collect()
        current_mem = get_process_memory_mb()
        checkpoints.append(current_mem)
        print(f"pass {i:02d}/{iterations:02d} | rss: {current_mem:.2f} mb | delta: {current_mem - baseline_mem:+.2f} mb")

    first_pass_mem = checkpoints[0]
    final_pass_mem= checkpoints[-1]
    drift = final_pass_mem - first_pass_mem

    print("\nmemory stability summary:")
    print(f"baseline post-load ram : {baseline_mem:.2f} mb")
    print(f"ram after pass 1       : {first_pass_mem:.2f} mb")
    print(f"ram after pass {iterations}      : {final_pass_mem:.2f} mb")
    print(f"drift across passes    : {drift:+.2f} mb")

    if drift < 25.0:
       print("\n[status: passed] memory footprint is stable. no leaks detected.")
    else:
        print("\n[status: failed] persistent memory growth detected. check graph cache.")   

if __name__ == "__main__":
    run_test_leak(iterations=20)    