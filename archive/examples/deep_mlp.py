"""
Deep ReLU MLP profiling demo.

Architecture (4 hidden layers, no bias):
  X(batch, 64) -> Linear(64→256) -> ReLU
               -> Linear(256→512) -> ReLU
               -> Linear(512→512) -> ReLU
               -> Linear(512→256) -> ReLU
               -> Linear(256→10)  -> MSELoss -> L

Compared to the two-layer MLP this model:
  - Has more activation tensors with longer lifetimes
  - Has more parameter gradients held until the optimizer step
  - Produces more Adam optimizer state tensors (exp_avg + exp_avg_sq x 5 layers)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from profiler import ProfilerExecutor


class DeepMLP(nn.Module):
    """4-hidden-layer MLP with ReLU activations, no bias."""

    def __init__(self, in_dim: int = 64, out_dim: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 512, bias=False)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 512, bias=False)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 256, bias=False)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(256, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        return self.fc5(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DeepMLP(in_dim=64, out_dim=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 32
    X = torch.randn(batch_size, 64, device=device)
    Y = torch.randn(batch_size, 10, device=device)

    profiler = ProfilerExecutor(model, optimizer, device)
    profiler.run(X, Y, loss_fn)

    report = profiler.get_report()
    report.print_summary()

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "deep_mlp_memory.png"
    )
    profiler.visualize(output_path=output_path, mode="static")


if __name__ == "__main__":
    main()
