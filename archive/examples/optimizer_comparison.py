"""
Optimizer memory comparison demo.

Runs the same two-layer MLP with three optimizers:
  - SGD (no momentum)       → no optimizer state
  - SGD with momentum=0.9   → one state tensor per parameter (momentum buffer)
  - Adam                    → two state tensors per parameter (exp_avg, exp_avg_sq)
                               plus a scalar step counter per parameter

Side-by-side summary shows how optimizer choice affects peak memory footprint.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from profiler import ProfilerExecutor
from profiler.graph import TensorRole


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 8, out_dim: int = 2) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden, bias=False)
        self.sig1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden, out_dim, bias=False)
        self.sig2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sig2(self.linear2(self.sig1(self.linear1(x))))


def run_one(optimizer_name: str, optimizer_cls, optimizer_kwargs: dict, output_dir: str):
    """Profile one training step with the given optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoLayerMLP(in_dim=4, hidden=8, out_dim=2).to(device)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    loss_fn = nn.MSELoss()

    torch.manual_seed(0)
    X = torch.randn(16, 4, device=device)
    Y = torch.randn(16, 2, device=device)

    profiler = ProfilerExecutor(model, optimizer, device)
    profiler.run(X, Y, loss_fn)

    report = profiler.get_report()

    roles = {role: report.registry.all_by_role(role) for role in TensorRole}
    total_bytes = sum(
        m.nbytes for tensors in roles.values() for m in tensors
    )

    print(f"\n{'='*60}")
    print(f"Optimizer: {optimizer_name}")
    print(f"{'='*60}")
    for role, tensors in roles.items():
        if tensors:
            kb = sum(m.nbytes for m in tensors) / 1024
            print(f"  {role.name:20} {len(tensors):3} tensors  {kb:7.2f} KB")
    print(f"  {'TOTAL':20}      {total_bytes / 1024:7.2f} KB")

    path = os.path.join(output_dir, f"memory_{optimizer_name.lower().replace(' ', '_')}.png")
    profiler.visualize(output_path=path, mode="static")
    return report


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    configs = [
        ("SGD (no momentum)",  torch.optim.SGD,  {"lr": 1e-3}),
        ("SGD + momentum",     torch.optim.SGD,  {"lr": 1e-3, "momentum": 0.9}),
        ("Adam",               torch.optim.Adam, {"lr": 1e-3}),
    ]

    reports = {}
    for name, cls, kwargs in configs:
        reports[name] = run_one(name, cls, kwargs, output_dir)

    print("\n\nSummary: optimizer state overhead")
    print(f"{'Optimizer':<22} {'Opt-state tensors':>18} {'Opt-state KB':>14}")
    print("-" * 58)
    for name, report in reports.items():
        states = report.registry.all_by_role(TensorRole.OPTIMIZER_STATE)
        kb = sum(m.nbytes for m in states) / 1024
        print(f"  {name:<20} {len(states):>18} {kb:>14.2f}")


if __name__ == "__main__":
    main()
