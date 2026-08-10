import torch

def calculate_entropy(probabilities: torch.Tensor, eps: float = 1e-12) -> float:
    clamped_probs = torch.clamp(probabilities, min=eps)
    entropy = -torch.sum(clamped_probs * torch.log2(clamped_probs))
    return entropy.item()


if __name__ == "__main__":
    # high certainty case (model is almost 100% confident)
    confident_probs = torch.tensor([0.99, 0.005, 0.003, 0.001, 0.001])
    h_confident = calculate_entropy(confident_probs)

    # high uncertainty case (model is confused between options)
    uncertain_probs = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20])
    h_uncertain = calculate_entropy(uncertain_probs)

    print(f"Confident Case Entropy : {h_confident:.4f} bits")
    print(f"Uncertain Case Entropy : {h_uncertain:.4f} bits")