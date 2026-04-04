"""
Benchmark harness for profiling real models.

Usage:
    python benchmarks.py                    # default: ResNet18
    python benchmarks.py Resnet50           # specify model
    python benchmarks.py Transformer        # Transformer model
    python benchmarks.py Bert               # BERT model (requires: pip install transformers)
"""

import sys
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fx as fx
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)
from torchvision.models import resnet18, resnet50
from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile
from visualizer import MemoryVisualizer
from activation_checkpoint import (
    select_activations_to_recompute,
    print_ac_decisions,
)


model_names: List[str] = [
    "Transformer",
    "Resnet18",
    "Resnet50",
    "Bert",
]

model_batch_sizes: Dict[str, int] = {
    "Transformer": 4,
    "Resnet18": 16,
    "Resnet50": 4,
    "Bert": 4,
}


class Experiment:
    """Benchmark harness for a single model.  Handles model creation, optimizer
    setup, training step definition, and the graph transformation callback that
    runs the profiler and AC selection."""

    def __init__(self, model_name: str, batch_size: int):
        assert model_name in model_names, (
            f"Model {model_name} not found in {model_names}"
        )
        dev = torch.device("cuda")
        self.model_name = model_name
        self.batch_size = batch_size

        if self.model_name == "Transformer":
            vocab_size = 2048
            bsz, seq_len = self.batch_size, 256
            with torch.device(dev):
                model_args = ModelArgs(
                    n_layers=8,
                    n_heads=4,
                    vocab_size=vocab_size,
                    max_seq_len=seq_len,
                    dropout_p=0.1,
                )
                self.model = Transformer(model_args)
            src = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            tgt = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            self.example_inputs = (src, tgt)

            def transformer_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any,
            ):
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = transformer_train_step
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=1e-2,
                foreach=True, capturable=True,
            )

        elif self.model_name in ["Resnet18", "Resnet50"]:
            inp = torch.randn(self.batch_size, 3, 224, 224, device=dev)
            num_classes = 10
            target = torch.randint(0, num_classes, (self.batch_size,), device=dev)
            self.example_inputs = (inp, target)
            with torch.device(dev):
                self.model = (
                    resnet18() if self.model_name == "Resnet18" else resnet50()
                )

            def resnet_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any,
            ):
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer = optim.Adam(
                self.model.parameters(), lr=1e-2,
                foreach=True, capturable=True,
            )
            self.train_step = resnet_train_step

        elif self.model_name == "Bert":
            from transformers import BertConfig, BertForSequenceClassification

            num_classes = 2
            seq_len = 128
            config = BertConfig(
                vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=512,
                num_labels=num_classes,
                # Disable features that break FX tracing.
                torchscript=True,
            )
            self.model = BertForSequenceClassification(config).to(dev)

            input_ids = torch.randint(
                0, config.vocab_size, (self.batch_size, seq_len), device=dev,
            )
            attention_mask = torch.ones(
                self.batch_size, seq_len, dtype=torch.long, device=dev,
            )
            labels = torch.randint(
                0, num_classes, (self.batch_size,), device=dev,
            )
            self.example_inputs = (input_ids, attention_mask, labels)

            def bert_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any,
            ):
                input_ids, attention_mask, labels = (
                    example_inputs[0],
                    example_inputs[1],
                    example_inputs[2],
                )
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer = optim.Adam(
                self.model.parameters(), lr=1e-2,
                foreach=True, capturable=True,
            )
            self.train_step = bert_train_step

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
        )

    def init_opt_states(self) -> None:
        """Run one dummy optimizer step to initialise Adam's lazy state."""
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def graph_transformation(
        self, gm: fx.GraphModule, args: Any,
    ) -> fx.GraphModule:
        print(f"\n{'=' * 80}")
        print(f"PROFILING: {self.model_name} (batch_size={self.batch_size})")
        print(f"{'=' * 80}")

        profiler = GraphProfiler(gm)
        warm_up_iters, profile_iters = 2, 3

        with torch.no_grad():
            for _ in range(warm_up_iters):
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(profile_iters):
                profiler.run(*args)

        profiler.aggregate_stats()
        profiler.print_stats()

        # Memory breakdown chart.
        chart_name = f"memory_{self.model_name.lower()}.png"
        viz = MemoryVisualizer(profiler)
        viz.plot_memory_timeline(
            chart_name,
            title=f"Memory Breakdown — {self.model_name} (bs={self.batch_size})",
        )

        # AC selection.
        to_recompute, to_retain = select_activations_to_recompute(profiler)
        print_ac_decisions(profiler, to_recompute, to_retain)

        return gm

    def run(self) -> None:
        """Initialise optimizer state, compile, and run the full pipeline."""
        self.init_opt_states()
        compiled_fn = compile(self.train_step, self.graph_transformation)
        compiled_fn(self.model, self.optimizer, self.example_inputs)
        print(f"{self.model_name} completed successfully.")


if __name__ == "__main__":
    # Accept model name as command-line argument, default to Resnet18.
    name = sys.argv[1] if len(sys.argv) > 1 else "Resnet18"
    if name not in model_names:
        print(f"Unknown model '{name}'. Choose from: {model_names}")
        sys.exit(1)
    exp = Experiment(name, model_batch_sizes[name])
    exp.run()
