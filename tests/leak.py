import os
import gc
import psutil
import torch
from model_loader import load_model_and_tokenizer
from generator import generate_stream

def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_leak_test(iterations: int = 15):
    print("[-] Initializing model and tokenizer...")
    start_mem = get_process_memory_mb()
    model, tokenizer = load_model_and_tokenizer()
    model_loaded_mem = get_process_memory_mb()
    
    print(f"[+] Model baseline memory: {model_loaded_mem:.2f} MB (Delta: +{model_loaded_mem - start_mem:.2f} MB)")
    print(f"[-] Executing {iterations} consecutive streaming cycles...\n")

    test_prompt = "Explain why the sky is blue in three concise sentences."

    memory_checkpoints = []

    for i in range(1, iterations + 1):
        stream = generate_stream(
            prompt=test_prompt,
            max_new_tokens=40,
            model=model,
            tokenizer=tokenizer,
            temperature=0.0,
            top_k=50,
            window_size=20
        )
        
        # consume the entire token stream
        for _ in stream:
            pass

        # run garbage collection checkpoint
        gc.collect()
        current_mem = get_process_memory_mb()
        memory_checkpoints.append(current_mem)
        print(f"Iteration {i:02d}/{iterations:02d} | Current RSS: {current_mem:.2f} MB | Delta from baseline: {current_mem - model_loaded_mem:+.2f} MB")

    initial_iter_mem = memory_checkpoints[0]
    final_iter_mem = memory_checkpoints[-1]
    net_growth = final_iter_mem - initial_iter_mem

    print("\n--- Leak Test Summary ---")
    print(f"Baseline Post-Model RAM: {model_loaded_mem:.2f} MB")
    print(f"Memory after 1st prompt: {initial_iter_mem:.2f} MB")
    print(f"Memory after {iterations}th prompt: {final_iter_mem:.2f} MB")
    print(f"Net Growth across iterations: {net_growth:+.2f} MB")

    # Threshold check: small variations under 15-20 MB are standard allocator overhead
    if net_growth < 25.0:
        print("\n[SUCCESS] No memory leak detected. RAM usage is stable.")
    else:
        print("\n[WARNING] Memory growth detected. Investigate dangling tensor references.")

if __name__ == "__main__":
    run_leak_test(iterations=15)