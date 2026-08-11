import torch
import math
from collections import deque
from typing import Dict

def calculate_entropy(probabilities: torch.Tensor, eps: float = 1e-12) -> float:
    # clamp probabilities to avoid log(0) undefined behavior
    clamped_probs = torch.clamp(probabilities, min=eps)
    entropy = -torch.sum(clamped_probs * torch.log2(clamped_probs))
    return entropy.item()

def calculate_delta_p(p1: float, p2: float) -> float:
    # calculate probability difference between top two candidates
    return p1 - p2

class EntropyTracker:
    def __init__(self, window_size: int = 20, eps: float = 1e-6):
        self.window = deque(maxlen=window_size)
        self.eps = eps

    def update(self, current_entropy: float) -> Dict[str, float]:
        self.window.append(current_entropy)
        
        n = len(self.window)
        mean = sum(self.window) / n
        
        if n < 2:
            std = 0.0
            z_score = 0.0
        else:
            variance = sum((x - mean) ** 2 for x in self.window) / (n - 1)
            std = math.sqrt(variance)
            z_score = (current_entropy - mean) / (std + self.eps)

        return {
            "entropy": current_entropy,
            "mean": mean,
            "std": std,
            "z_score": z_score
        }


if __name__ == "__main__":
    tracker = EntropyTracker(window_size=5)
    
    # simulated entropy values with a sudden jump at the end
    test_entropies = [0.10, 0.12, 0.11, 0.15, 2.45]
    
    print("***** Moving Z-Score Test *****")
    for h in test_entropies:
        stats = tracker.update(h)
        print(f"H: {stats['entropy']:.2f} | Mean: {stats['mean']:.2f} | Std: {stats['std']:.2f} | Z-Score: {stats['z_score']:.2f}")