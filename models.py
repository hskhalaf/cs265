"""
useful notes:

- each `make_*` returns a (model, optimizer, example_inputs, train_step) tuple
- train_step(model, optim, example_inputs) runs forward + loss + backward + optimizer.step + zero_grad
- the loss is wrapped in SEPFunction.apply()
- every optimizer is Adam with `foreach=True, capturable=True` so the optimizer step traces into a sequence of `_foreach_*` ops.
"""

from typing import Any, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_tracer import SEPFunction
from torchvision.models import resnet18, resnet50
from transformers import BertConfig, BertForSequenceClassification

ModelTuple = Tuple[nn.Module, torch.optim.Optimizer, Tuple[Any, ...], Callable]



class DummyModel(nn.Module):
    def __init__(self, layers: int = 10, dim: int = 100):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dummy(batch_size: int = 1000, layers: int = 10, dim: int = 100, device: str = "cuda") -> ModelTuple:
    model = DummyModel(layers=layers, dim=dim).to(device)
    batch = torch.randn(batch_size, dim, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True, capturable=True)
    # foreach=True decomposes Adam's step into a chain of _foreach_* operations during tracing, 
    # which is what produces the foreach-getitem patterns the profiler handles. 
    # capturable=True keeps the step counter as a tensor instead of a Python int
    def train_step(model, optim, example_inputs):
        (x,) = example_inputs
        loss = model(x).sum()
        loss = SEPFunction.apply(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

    return model, optim, (batch,), train_step


def _make_resnet(arch: str, batch_size: int, num_classes: int, device: str) -> ModelTuple:
    builder = {"resnet18": resnet18, "resnet50": resnet50}[arch]
    with torch.device(device):
        model = builder(num_classes=num_classes)
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True, capturable=True)

    def train_step(model, optim, example_inputs):
        x, y = example_inputs
        loss = F.cross_entropy(model(x), y)
        loss = SEPFunction.apply(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

    return model, optim, (inputs, targets), train_step

def make_resnet18(batch_size: int = 16, device: str = "cuda") -> ModelTuple:
    return _make_resnet("resnet18", batch_size, num_classes=10, device=device)

def make_resnet50(batch_size: int = 16, device: str = "cuda") -> ModelTuple:
    return _make_resnet("resnet50", batch_size, num_classes=10, device=device)



def make_bert(batch_size: int = 4, seq_len: int = 128, device: str = "cuda") -> ModelTuple:

    num_classes = 2
    config = BertConfig(
        vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=512, num_labels=num_classes,
        torchscript=True,                # FX-friendly
    )
    model = BertForSequenceClassification(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True, capturable=True)

    def train_step(model, optim, example_inputs):
        ids, m, y = example_inputs
        outputs = model(input_ids=ids, attention_mask=m, labels=y)
        loss = SEPFunction.apply(outputs.loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

    return model, optim, (input_ids, mask, labels), train_step


MODELS: dict[str, Callable[..., ModelTuple]] = {"dummy": make_dummy, "resnet18": make_resnet18, "resnet50": make_resnet50, "bert": make_bert,}

def init_optimizer_state(model: nn.Module, optim: torch.optim.Optimizer) -> None:
    """
    run one dummy step so Adam allocates its moment buffers.
    reason: the graph tracer needs the optimizer state to already exist when it snapshots the optimizer; 
    without this, Adam's lazy init fires inside the traced graph and the moment buffers don't appear as placeholders.
    """
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.rand_like(p)
    optim.step()
    optim.zero_grad()
