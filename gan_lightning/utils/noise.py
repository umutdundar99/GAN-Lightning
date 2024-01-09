import torch


def create_noise(n_samples: int, input_dim: int, device: torch.device = "cuda:0"):
    return torch.randn(n_samples, input_dim, device=device)
