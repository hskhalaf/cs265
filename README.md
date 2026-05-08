# CS265 — Activation Checkpointing in PyTorch

An implementation of the µ-TWO activation-checkpointing pipeline on top of
`torch.fx`.  The repo is structured to be readable end-to-end: nine source
files, every one under 300 lines.

## Pipeline at a glance

```
train_step  ──compile──▶  FX GraphModule  ──┐
                                            ▼
                                    GraphProfiler  (Phase 1)
                                    ├─ static analysis (regions, roles, lifetimes)
                                    └─ runtime profiling (per-node timing + memory)
                                            ▼
                                    select_activations  (Phase 2)
                                    └─ µ-TWO greedy on size/recompute_time ratio
                                            ▼
                                    rewrite_with_checkpointing  (Phase 3)
                                    └─ splice forward subgraphs into the backward pass
                                            ▼
                                    rewritten GraphModule  ──run──▶
```

## Files

| File | Purpose |
|---|---|
| [graph_tracer.py](graph_tracer.py) | Course-provided.  `compile(fn, transform)` traces a stateless training step into a single FX graph and runs the optional transform on first call. |
| [utils.py](utils.py) | Course-provided.  `SPMD_DECOMP_TABLE` for tracing fused/foreach optimizer ops. |
| [graph_prof.py](graph_prof.py) | **Phase 1.**  `GraphProfiler` — static analysis + per-node timing + storage-aware memory accounting. |
| [activation_checkpoint.py](activation_checkpoint.py) | **Phases 2 + 3.**  `select_activations` (µ-TWO greedy), `rewrite_with_checkpointing` (graph rewriter). |
| [visualizer.py](visualizer.py) | All plotting — memory breakdown stacked-area chart and deliverable grouped-bar charts. |
| [models.py](models.py) | Single source of truth for `(model, optimizer, example_inputs, train_step)` across DummyModel / ResNet-18 / ResNet-50 / BERT-base. |
| [starter_code.py](starter_code.py) | Read-this-first entry point.  Runs the full pipeline on DummyModel. |
| [validate.py](validate.py) | Three end-to-end sanity checks: profiler accuracy, AC sanity per architecture, AC correctness (rewritten ↔ original outputs match). |
| [benchmarks.py](benchmarks.py) | Sweep harness.  Produces `plots/peak_<model>.png` and `plots/latency_<model>.png` for the deliverable. |

## Install

Python 3.12, CUDA-capable GPU.

```bash
pip install -r requirements.txt          # torch, torchvision, transformers, matplotlib, numpy
```

## Run

```bash
python starter_code.py                    # full pipeline on DummyModel; produces memory_breakdown.png
python validate.py --all                  # sanity checks across all four models
python benchmarks.py                      # full sweep -> plots/peak_*.png + plots/latency_*.png
python benchmarks.py --quick              # smaller batch ranges for a quick smoke test
python benchmarks.py resnet18             # one model only
```

## What each phase produces

### Phase 1 — Profiling (`graph_prof.py`)

For every node in the traced graph the profiler records:

- region (FORWARD / LOSS / BACKWARD / OPTIMIZER)
- tensor role (PARAM / ACT / GRAD / OPT_STATE / OPT_SCRATCH / OTHER)
- average runtime (CUDA-event timed, averaged over multiple iterations)
- size in bytes — **storage-aware**: ops whose schema declares output→input
  aliasing (in-place `_foreach_*`, `_fused_adam`, view ops) contribute zero
  new bytes, which prevents the optimizer-step "double-counting spike" you
  see in naive implementations.

Activations (`NodeType.ACT`) are forward call_function nodes whose output is
read in the backward pass — precisely the candidates AC can recompute.

### Phase 2 — Selection (`activation_checkpoint.select_activations`)

µ-TWO greedy: at each step evict the activation with the highest
`size_bytes / recompute_ms` ratio (best memory return per millisecond), then
re-simulate peak with that eviction set.  Stops when the target peak is hit
or when an eviction round fails to lower the peak (peak isn't dominated by
activations — typical for BERT, where Adam's 2× moment buffers dominate).

Cascading recomputation: when an upstream activation is evicted, the cost of
recomputing any downstream activation that depends on it grows by the
upstream's recompute time.

### Phase 3 — Rewriting (`activation_checkpoint.rewrite_with_checkpointing`)

For each evicted activation, in forward order:

1. Walk back from the activation, stopping at the boundary of nodes
   guaranteed live at backward time (placeholders + retained intermediates +
   already-recomputed activations).
2. Extract the forward subgraph between those boundary nodes and the
   activation, using `torch._functorch.partitioners._extract_graph_with_inputs_outputs`.
3. Splice a copy of the subgraph into the backward pass right before the
   activation's first backward use.
4. Redirect later backward uses of the original activation to the
   recomputed copy.

The rewritten graph still runs end-to-end and produces gradients identical
(within float-rounding) to the original — `validate.py` Check 3 verifies
this.

## Output

```
memory_breakdown.png        from starter_code.py — stacked area, role-coloured
plots/peak_<model>.png      from benchmarks.py  — peak memory  vs batch size, AC off vs on
plots/latency_<model>.png   from benchmarks.py  — iter latency vs batch size, AC off vs on
```
