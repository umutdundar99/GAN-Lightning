import torch


def create_noise(n_samples: int, z_dim: int, device: torch.device = "cuda:0"):
    return torch.randn(n_samples, z_dim, device=device)
