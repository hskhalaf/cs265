"""
Two-layer MLP profiling demo.

Architecture matching the spec:
  X  -> Linear(W1) -> Z1 -> Sigmoid -> Z2
      -> Linear(W3) -> Z3 -> Sigmoid -> Z4 -> MSELoss(Y) -> L

Backward computes: grad_W1, grad_W3
Optimizer updates: W1, W3
"""
import sys
import os

# Allow running from the cs265 root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from profiler import ProfilerExecutor


class TwoLayerMLP(nn.Module):
    """Two linear layers with sigmoid activations, no bias."""

    def __init__(self, in_dim: int = 4, hidden: int = 8, out_dim: int = 2) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden, bias=False)   # W1
        self.sig1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden, out_dim, bias=False)  # W3
        self.sig2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.linear1(x)   # Z1 = X @ W1
        z2 = self.sig1(z1)     # Z2 = sigmoid(Z1)
        z3 = self.linear2(z2)  # Z3 = Z2 @ W3
        z4 = self.sig2(z3)     # Z4 = sigmoid(Z3)
        return z4


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TwoLayerMLP(in_dim=4, hidden=8, out_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 16
    X = torch.randn(batch_size, 4, device=device)
    Y = torch.randn(batch_size, 2, device=device)

    profiler = ProfilerExecutor(model, optimizer, device)
    profiler.run(X, Y, loss_fn)

    report = profiler.get_report()
    report.print_summary()

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "memory_breakdown.png"
    )
    profiler.visualize(output_path=output_path, mode="static")


if __name__ == "__main__":
    main()
