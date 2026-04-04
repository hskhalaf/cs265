"""Quick debug script to inspect the traced FX graph structure."""
import torch
import torch.nn as nn
import torch.fx as fx
from graph_tracer import SEPFunction, compile as gt_compile
from typing import Any


class DummyModel(nn.Module):
    def __init__(self, layers=2, dim=16):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


def debug_transformation(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
    print("=" * 80)
    print("GRAPH NODES")
    print("=" * 80)
    for i, node in enumerate(gm.graph.nodes):
        val = node.meta.get("val", None)
        rg = None
        if isinstance(val, torch.Tensor):
            rg = val.requires_grad
        print(
            f"{i:>3} op={node.op:<16} name={node.name:<30} "
            f"target={str(node.target)[:50]:<50} "
            f"requires_grad={rg}  "
            f"users={[u.name for u in node.users]}"
        )
    print("=" * 80)
    return gm


def main():
    device = "cuda:0"
    model = DummyModel(layers=2, dim=16).to(device)
    batch = torch.randn(8, 16, device=device)
    opt = torch.optim.Adam(
        model.parameters(), lr=0.01, foreach=True, capturable=True,
    )
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.rand_like(p)
    opt.step()
    opt.zero_grad()

    def train_step(model, optim, batch):
        loss = model(batch).sum()
        loss = SEPFunction.apply(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

    compiled_fn = gt_compile(train_step, debug_transformation)
    compiled_fn(model, opt, batch)


if __name__ == "__main__":
    main()
