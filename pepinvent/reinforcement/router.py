import torch
import torch.nn as nn
from torch.distributions import Categorical

class Router(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.phi = nn.Parameter(torch.zeros(L))

    def forward(self):
        """Return categorical probabilities over L positions."""
        return torch.softmax(self.phi, dim=-1)

    def sample_mask(self, num_position=1, num_sample=1):
        """
        Sample `num_position` positions for each batch element.

        Args:
            p (torch.Tensor, optional): Probabilities over L positions. Defaults to self.forward().
            num_position (int): Number of positions to sample.
            unique (bool): If True, samples without replacement.
            num_sample (int): Number of independent draws (batches).

        Returns:
            torch.LongTensor of shape (num_sample, num_position)
                if num_position > 1, else (num_sample,)
        """
        p = self.forward()  # shape (L,)

        if num_position > 1:
            # Without replacement
            idx = torch.stack([torch.multinomial(p, num_samples=num_position, replacement=False)
                               for _ in range(num_sample)])
        else:
            # With replacement (independent samples)
            dist = Categorical(p)
            if num_position == 1:
                idx = dist.sample((num_sample,))  # (num_sample,)
            else:
                idx = dist.sample((num_sample, num_position))  # (num_sample, num_position)

        return idx

    def log_prob(self, idx, reduce: str = "sum"):
        """
        Compute log-probabilities under current policy.

        Args:
            idx: Tensor of indices (num_sample,) or (num_sample, num_position)
            reduce: "none" | "sum" | "mean"
        """
        p = self.forward()
        logp = torch.log(p.clamp(min=1e-8))
        idx = torch.as_tensor(idx, dtype=torch.long, device=logp.device)
        gathered = logp.index_select(0, idx.reshape(-1)).reshape(idx.shape)

        if reduce == "none":
            return gathered
        elif reduce == "sum":
            return gathered.sum(dim=-1)
        elif reduce == "mean":
            return gathered.mean(dim=-1)
        else:
            raise ValueError(f"Invalid reduce='{reduce}'. Use 'none' | 'sum' | 'mean'.")

    def entropy(self, p: torch.Tensor = None) -> torch.Tensor:
        """Entropy H(p) in nats."""
        if p is None:
            p = self.forward()
        p_safe = p.clamp(min=1e-12)
        return -(p_safe * torch.log(p_safe)).sum()