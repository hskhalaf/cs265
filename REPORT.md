# CS265 Activation Checkpointing — Technical Report

**Hadi Khalaf, CS265, Spring 2026**

## Table of Contents
1. [Introduction](#1-introduction)
2. [Background: FX Tracing and the Compiled Graph](#2-background-fx-tracing-and-the-compiled-graph)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Phase 1: Graph Profiler](#4-phase-1-graph-profiler)
   - 4.1 [Static Analysis](#41-static-analysis)
   - 4.2 [Runtime Profiling](#42-runtime-profiling)
   - 4.3 [Memory Visualisation](#43-memory-visualisation)
5. [Phase 2: Activation Checkpointing Selection](#5-phase-2-activation-checkpointing-selection)
   - 5.1 [The Problem](#51-the-problem)
   - 5.2 [The mu-TWO Greedy Algorithm](#52-the-mu-two-greedy-algorithm)
   - 5.3 [Memory Simulator](#53-memory-simulator)
   - 5.4 [Cascading Recomputation](#54-cascading-recomputation)
   - 5.5 [Validation of Recompute Decisions](#55-validation-of-recompute-decisions)
6. [Experimental Results](#6-experimental-results)
   - 6.1 [DummyModel](#61-dummymodel)
   - 6.2 [ResNet18](#62-resnet18)
   - 6.3 [ResNet50](#63-resnet50)
   - 6.4 [BERT](#64-bert)
   - 6.5 [Validation](#65-validation)
7. [Design Decisions and Challenges](#7-design-decisions-and-challenges)
8. [Interpreting the Output](#8-interpreting-the-output)
9. [Known Limitations and Simplifications](#9-known-limitations-and-simplifications)
10. [Test Suite](#10-test-suite)
    - 10.1 [Testing philosophy](#101-testing-philosophy)
    - 10.2 [Test categories](#102-test-categories)
    - 10.3 [Running the tests](#103-running-the-tests)

---

## 1. Introduction

Training a deep neural network requires far more GPU memory than the model weights alone.  During a single training iteration the GPU must simultaneously hold parameters, activations (intermediate outputs of each layer), gradients, and optimizer state (Adam's two moment buffers per parameter).  Activations account for 70-85% of peak memory because they are produced in the forward pass but cannot be freed until their corresponding backward computation consumes them — and the two passes run in opposite order, so the earliest activations live the longest.

**Activation checkpointing** addresses this by discarding selected activations after the forward pass and recomputing them from retained values during the backward pass, trading extra computation for a lower memory peak.  This project implements the infrastructure required to make that trade-off intelligently:

- **Phase 1 (Graph Profiler)** traces one full training step into a single FX graph, executes it node-by-node to collect per-operator timing and memory statistics, classifies every tensor into five categories (parameter, activation, gradient, optimizer state, other), computes activation lifetimes, and produces a peak memory breakdown chart.

- **Phase 2 (AC Selection Algorithm)** implements the mu-TWO greedy algorithm (Purandare et al., MLSys 2023) with a memory simulator that re-estimates peak memory after each eviction decision, cascading recomputation cost tracking, and early termination when the peak is not reducible by activation checkpointing.

Phase 3 (graph rewriting to insert recomputation nodes) will be implemented next.

We evaluate the system on four models: DummyModel (10-layer MLP), ResNet18, ResNet50, and BERT-base.  On ResNet18, the algorithm reduces peak memory from 506 MB to 431 MB by evicting 8 cheap intermediates at a cost of 1 ms.  On ResNet50, peak drops from 684 MB to 585 MB.  On BERT, the algorithm correctly identifies that the peak is in the optimizer region where activation checkpointing cannot help, and retains all 363 intermediates.

---

## 2. Background: FX Tracing and the Compiled Graph

### Why FX?

PyTorch's `torch.fx` framework captures a Python function into a symbolic `GraphModule` — a DAG of operator-level nodes with explicit data dependencies.  Unlike module-level hooks (which observe execution from the outside and must infer the graph), FX tracing produces a complete, rewritable graph at compile time.  This is essential for Phase 3: you cannot insert recomputation nodes into a graph you cannot rewrite.

### How the graph is produced

The `graph_tracer.compile()` function does the following:

1. Extract the `nn.Module` and `Optimizer` from the arguments.
2. Lift all parameters, buffers, and optimizer states into function arguments so that `make_fx()` can trace operations applied to them.
3. Trace the training step (forward + loss + backward + optimizer) in `FakeTensorMode`, which performs shape inference without moving real data.
4. Apply a decomposition table (`SPMD_DECOMP_TABLE`) that expands fused operators (like `_fused_adam`) into primitive operations.
5. Remove identity `detach` and `tag_grad` nodes inserted by autograd.
6. Return a flat `GraphModule` whose inputs are `[params, buffers, opt_states, batch]` and whose graph contains every operation from the training step.

### Separator nodes

The training step wraps the loss with `SEPFunction.apply()`, which inserts two identity operations into the traced graph:

- `%sep` — marks the end of the forward pass.
- `%sep_backward` — marks the beginning of the backward pass.

Everything before `%sep` is the forward pass.  Everything between `%sep` and `%sep_backward` is the loss computation.  Everything after `%sep_backward` is the backward pass.  The optimizer step follows at the end.

### Graph node structure

Each node in the FX graph has:

- `node.op` — the operation type (`placeholder`, `call_function`, `output`).
- `node.target` — the ATen operator (e.g. `torch.ops.aten.mm.default`, `torch.ops.aten.relu.default`).
- `node.all_input_nodes` — the nodes that produce this node's inputs.
- `node.users` — the nodes that consume this node's output.
- `node.meta['val']` — a `FakeTensor` recording the output shape and dtype (from symbolic tracing).

Since the nodes are already in topological order, we can walk them with a single linear pass.

---

## 3. High-Level Architecture

```
graph_tracer.py         Provided by course.  Traces train_step into FX graph.
graph_prof.py           Phase 1.  Static analysis + runtime profiling.
activation_checkpoint.py  Phase 2.  mu-TWO greedy selection + memory simulator.
visualizer.py           Memory breakdown stacked bar chart.
starter_code.py         Entry point: profile -> visualise -> select -> (rewrite).
benchmarks.py           ResNet18/50, Transformer, BERT benchmarks.
validate.py             Three-level validation: profiler accuracy, AC sanity,
                        memory simulator consistency.
utils.py                Provided by course.  SPMD decomposition table.
tests/test_profiler.py  55-test suite across 8 categories.
```

The execution flow is:

```
starter_code.experiment()
  |-- Build DummyModel, Adam optimizer
  |-- Initialise optimizer state (Adam lazy init)
  +-- compile(train_step, graph_transformation)
        |-- graph_tracer._compile() -> GraphModule
        +-- graph_transformation(gm, args)
              |-- GraphProfiler(gm)           <- static analysis
              |-- profiler.run(*args) x N     <- warm-up
              |-- profiler.reset_stats()
              |-- profiler.run(*args) x M     <- measurement
              |-- profiler.aggregate_stats()
              |-- profiler.print_stats()      <- report
              |-- MemoryVisualizer(profiler)
              |     +-- plot_memory_timeline() <- PNG chart
              |-- select_activations_to_recompute(profiler)
              +-- print_ac_decisions()        <- selection report
```

---

## 4. Phase 1: Graph Profiler

The profiler lives in `graph_prof.py` and extends `torch.fx.Interpreter`.  The `Interpreter` base class executes a `GraphModule` node-by-node, calling `run_node()` for each operation.  We override `__init__` for static analysis and `run_node` for runtime measurement.

### 4.1 Static Analysis

All static analysis happens in `__init__`, performing a single linear walk over the graph's nodes.

#### Step 1 — Locate boundary nodes

We scan for the two separator targets:

```python
for i, node in enumerate(self.node_list):
    if node.target == torch.ops.separator.sep.default:
        self.sep_node, self.sep_index = node, i
    elif node.target == torch.ops.separator.sep_backward.default:
        self.sep_bwd_node, self.sep_bwd_index = node, i
```

These two indices partition the graph into four regions: FORWARD (index <= sep), LOSS (between sep and sep_backward), BACKWARD (>= sep_backward), and OPTIMIZER (>= optimizer boundary).

#### Step 2 — Extract parameters, gradients, and optimizer states

Parameter identification uses two strategies depending on the optimizer configuration:

**Strategy A (`fused=True`).** The traced graph contains a single `_fused_adam` node whose arguments are structured lists:

```python
self.param_nodes = set(self.optimizer_node.args[0])  # parameter tensors
self.grad_nodes  = set(self.optimizer_node.args[1])  # gradient tensors
```

**Strategy B (`foreach=True`).** The optimizer step is decomposed into many individual `_foreach_*` and `copy_` operations — there is no single optimizer node.  We first detect the optimizer region boundary (the first `_foreach_*` op after `sep_backward`), then identify parameters by their usage pattern:

```python
has_fwd_user = any(idx < self.sep_index for idx in user_indices)
has_opt_user = any(idx >= self.optimizer_index for idx in user_indices)
if has_fwd_user and has_opt_user:
    self.param_nodes.add(node)
```

The key insight: **parameters are the only placeholder nodes that appear in both the forward pass and the optimizer step.**  Batch data appears in forward + backward but not the optimizer.  Optimizer states appear only in the optimizer region.

**Optimizer state classification.**  Placeholder nodes whose users are exclusively in the optimizer region are classified as `OPTIMIZER_STATE`.  This gives us the five categories required by the course spec: PARAM, ACT, GRAD, OPTIMIZER_STATE, and OTHER.

#### Step 3 — Identify intermediate activations

An intermediate activation satisfies four conditions:

1. It is a `call_function` node (a computation, not an input or output).
2. Its index is strictly before `sep_index` (produced during the forward pass).
3. It is not a parameter node.
4. It has at least one user whose index is at or after `sep_bwd_index` (consumed during the backward pass).

These are exactly the tensors that activation checkpointing can target.

#### Step 4 — Compute lifetimes

For each intermediate we record:

- **`last_fwd_access`** — maximum index among forward-region users.
- **`first_bwd_access`** — minimum index among backward-region users.

The difference `first_bwd_access - last_fwd_access` is the **lifetime** — the number of steps during which the tensor sits idle in GPU memory.

#### Step 5 — Compute tensor sizes

Each node's output shape and dtype are available from `node.meta['val']`:

```python
def _tensor_size_bytes(val):
    if isinstance(val, torch.Tensor):
        return val.numel() * val.element_size()
```

### 4.2 Runtime Profiling

The `run_node` override does three things for every node:

#### Timing

We bracket each node with `torch.cuda.Event` pairs and call `torch.cuda.synchronize()` to ensure GPU kernels complete before reading elapsed time.

#### Memory tracking

Before and after each node, we read `torch.cuda.memory_allocated()`.  The delta tells us the net GPU memory change.

#### Activation swapping

During profiling, intermediates are swapped to CPU after their last forward use and back to GPU before their first backward use.  This both measures actual swap overhead per tensor and prevents GPU OOM when profiling large models.

#### Aggregation

The profiler runs for `W` warm-up iterations followed by `M` measurement iterations.  `reset_stats()` clears all accumulators between phases.  `aggregate_stats()` averages measurements and populates `IntermediateInfo` fields used by Phase 2.

### 4.3 Memory Visualisation

`visualizer.py` produces a stacked bar chart showing live memory at each step, broken down by tensor role:

- **Blue** = PARAM
- **Green** = ACT (activations accumulate during forward, freed during backward)
- **Orange** = GRAD
- **Red** = OPTIMIZER_STATE
- **Purple** = OTHER

The peak step is marked with a dashed line; separator boundaries with dotted lines.

---

## 5. Phase 2: Activation Checkpointing Selection

### 5.1 The Problem

Given `N` intermediate activations, each with memory size `mem_i` and recomputation cost `cost_i`, choose which to retain and which to discard (recompute during backward).  The goal: reduce peak memory below a target `mem_limit` while minimising extra computation.  This is NP-hard in general.

### 5.2 The mu-TWO Greedy Algorithm

We implement Algorithm B from the mu-TWO paper (Purandare et al., MLSys 2023), adapted for single-model activation checkpointing (recompute only, no CPU offloading).  The ranking metric is the **recompute ratio**:

```
recompute_ratio = memory_size / total_recomp_time
```

The algorithm is iterative:

```
1.  Compute baseline peak via memory simulator.
2.  Set mem_limit = baseline_peak - 0.5 * total_activation_memory.
3.  Quick check: if evicting ALL intermediates doesn't reduce peak,
    AC cannot help — return immediately (retain everything).
4.  While candidate set is not empty:
    a. Simulate current peak. If peak <= mem_limit, stop.
    b. If last eviction didn't reduce peak, stop (early termination).
    c. Pick candidate with highest recompute_ratio.
    d. Evict it. Propagate dependency changes (see 5.4).
5.  Validate the recompute set (see 5.5).
```

The early termination in step 4b is critical: when the peak is in the optimizer region (as with BERT), no amount of activation eviction can reduce it.  Without this check, the algorithm would pointlessly evict all intermediates.

### 5.3 Memory Simulator

The memory simulator (`_simulate_peak_memory`) estimates peak memory given a set of evicted intermediates.  It walks the node timeline and computes live memory at each step:

- **Non-evicted tensors** are live from `produced_at` to `last_use`.
- **Evicted intermediates** are live during their forward period (`produced_at` to `last_fwd_access`), then freed, then briefly live at their recomputation point (`first_bwd_access`).

The simulator is called after each eviction decision to check whether the peak has dropped below `mem_limit`.  This is how the greedy loop knows when to stop — it directly measures the effect of each decision rather than relying on a static budget.

### 5.4 Cascading Recomputation

When tensor A is evicted and is a recomputation source (`recomp_src`) of another tensor B, B can no longer directly access A during its own recomputation.  B's recomputation would first need to recompute A, increasing B's total cost:

```python
if best_node in cand["recomp_srcs"]:
    cand["recomp_srcs"].discard(best_node)
    cand["recomp_srcs"].update(candidates[best_node]["recomp_srcs"])
    cand["recomp_cnt"] += candidates[best_node]["recomp_cnt"]
    cand["total_recomp_time"] = cand["recomp_time"] * cand["recomp_cnt"]
    cand["recomp_ratio"] = cand["memory_size"] / (cand["total_recomp_time"] + 1e-9)
```

This corresponds to Algorithms E and F in the mu-TWO paper.  The cascade propagates the evicted node's sources upward and increases the dependent node's `recomp_cnt`, which lowers its ratio and makes it less likely to be evicted next.

### 5.5 Validation of Recompute Decisions

Not every activation can be recomputed.  To recompute `x` during backward, every input to the op that produced `x` must be available — either a placeholder (always in memory) or a retained intermediate.  The validation pass iterates until stable:

```python
while changed:
    for node in list(recompute_set):
        inputs_ok = all(
            inp in valid_inputs or inp not in recompute_set
            for inp in node.all_input_nodes
        )
        if not inputs_ok:
            retain_set.add(node)
            recompute_set.discard(node)
            changed = True
```

---

## 6. Experimental Results

All experiments use Adam with `foreach=True` and `capturable=True`, 2 warm-up iterations and 3 measurement iterations.

### Summary

| Model | Intermediates | Act. Memory | Peak (est.) | Peak (actual) | Recomputed | Retained | Peak After AC | Extra Cost |
|-------|-------------:|------------:|------------:|--------------:|-----------:|---------:|--------------:|-----------:|
| DummyModel (10 layers) | 19 | 4.16 MB | 6.46 MB | 14.51 MB | 3 | 16 | 5.29 MB | 0.18 ms |
| ResNet18 (bs=16) | 103 | 331 MB | 506 MB | 523 MB | 8 | 95 | 431 MB | 1.05 ms |
| ResNet50 (bs=4) | 268 | 333 MB | 684 MB | — | 216 | 52 | 585 MB | 8.73 ms |
| BERT-base (bs=4) | 363 | 716 MB | 2506 MB | 2118 MB | 0 | 363 | 2506 MB | 0 ms |

The estimated peak is the formula-based static estimate; the actual peak is from `torch.cuda.max_memory_allocated()`.  The gap is due to CUDA allocator overhead, temporary buffers, and fragmentation (see Section 6.5).

### 6.1 DummyModel

DummyModel (10 layers of Linear + ReLU, dim=100, batch=1000) produces 19 intermediates totalling 4.16 MB.  The algorithm evicts 3 relu outputs (cheapest to recompute), reducing estimated peak from 6.46 MB to 5.29 MB.  The algorithm stops after just 3 evictions because the memory simulator reports the target has been met.

### 6.2 ResNet18

ResNet18 (batch size 16, 224x224 images) produces 103 intermediates totalling 331 MB.  The algorithm evicts **8 intermediates** (large pooling/relu outputs with high recompute ratios) and **retains 95** (all 19 convolutions plus smaller ops).  Peak drops from 506 MB to 431 MB at a cost of 1.05 ms.

The retained activations are exclusively convolution outputs — convolutions are the most expensive forward operations and have the lowest recompute ratios.  The recomputed activations are ReLU outputs and pooling results, which are nearly free to recompute.

Profiler accuracy: estimated 506 MB vs actual GPU 523 MB (ratio 0.97 — near-perfect match).

### 6.3 ResNet50

ResNet50 (batch size 4) produces 268 intermediates totalling 333 MB.  The algorithm evicts **216 intermediates** (all relus, getitems, transposes, and the initial large convolution) and retains **52** (all deeper convolutions).  Peak drops from 684 MB to 585 MB (99 MB reduction) at a cost of 8.73 ms.

The larger recompute set compared to ResNet18 reflects ResNet50's deeper architecture with many more batch norm and relu intermediates that are cheap to recompute.

### 6.4 BERT

BERT-base (12 layers, 768 hidden, 12 attention heads, seq_len=128, batch=4) produces 363 intermediates totalling 716 MB.  However, the peak memory (2506 MB) occurs at step 7366 in the **optimizer region**, where no forward-pass activations are live.

The algorithm's quick pre-check detects this: simulating peak with all intermediates evicted produces the same 2506 MB.  It correctly returns immediately with **0 recomputed, 363 retained**.

This is an important finding: **BERT's memory bottleneck under `foreach=True` Adam is the optimizer step, not the activations.**  The decomposed `_foreach_*` operations create many temporary tensors during the Adam update.  Activation checkpointing is the wrong tool here — reducing optimizer memory (via `fused=True`, gradient accumulation, or optimizer state sharding) would be more effective.

Profiler accuracy: estimated 2506 MB vs actual GPU 2118 MB (ratio 1.18 — the overestimate is because we count every FX node's FakeTensor size including optimizer intermediates that CUDA handles more efficiently via in-place operations).

### 6.5 Validation

The validation script (`validate.py`) runs three checks per model:

**Check 1 — Profiler accuracy.**  Compares our static peak estimate against `torch.cuda.max_memory_allocated()`.  Results:

| Model | Estimated | Actual | Ratio |
|-------|----------:|-------:|------:|
| DummyModel | 6.46 MB | 14.51 MB | 0.45 |
| ResNet18 | 506 MB | 523 MB | 0.97 |
| BERT | 2506 MB | 2118 MB | 1.18 |

The DummyModel gap (0.45) is expected: small tensors (40 KB each) incur proportionally large CUDA allocator rounding overhead.  For larger models the static estimate converges to the actual measurement.

**Check 2 — AC decision sanity.**  Full coverage (recompute + retain = all intermediates), no overlap, input availability, and model-specific checks (DummyModel: relus recomputed; ResNet18: convolutions retained; BERT: nothing recomputed when peak is in optimizer region).  All pass.

**Check 3 — Memory simulator consistency.**  All-evicted peak <= baseline; single eviction reduces peak by at most the tensor's size; monotonicity (evicting more never increases peak).  All pass.

---

## 7. Design Decisions and Challenges

### Decision 1: FX-based profiling instead of hooks

The Milestone 1 prototype used PyTorch module hooks.  FX tracing eliminates the need to reconstruct the graph, track tensor identity across callbacks, and infer implicit autograd saves.  The traced graph is a single explicit DAG with stable node names and exact data dependencies.

### Decision 2: Five tensor categories

The course spec requires classifying tensors as parameter, gradient, activation, optimizer state, or other.  We identify optimizer states as placeholder nodes whose users are exclusively in the optimizer region.  With `foreach=True` Adam, gradients cannot be separately identified (they are classified as OTHER), but this does not affect AC decisions which depend only on parameter and activation classification.

### Decision 3: Memory simulator as stopping criterion

The mu-TWO paper uses a memory simulator (Algorithm G) to validate each eviction decision.  Rather than using a fixed budget (e.g. "evict 50% of activation memory"), our implementation re-simulates peak memory after every eviction and stops when the peak drops below `mem_limit`.  This is more faithful to the paper and produces better results: it stops early when a few large evictions suffice and avoids pointless eviction when the peak is not activation-dominated.

### Decision 4: Early termination for non-reducible peaks

When the peak occurs in the optimizer or backward region where no forward-pass activations are live, AC cannot help.  We detect this with a quick pre-check (simulate peak with all intermediates evicted) and return immediately.  Without this, BERT would wastefully evict all 363 intermediates with zero peak reduction.

### Challenge 1: Parameter identification without _fused_adam

`fused=True` Adam crashes during FX tracing on some PyTorch versions.  The course starter code uses `foreach=True`, which decomposes the optimizer into hundreds of `_foreach_*` ops.  FakeTensor metadata has `requires_grad=False` for all nodes.  We identify parameters by their usage pattern: placeholder nodes with users in both forward and optimizer regions.

### Challenge 2: Cascading recomputation cost

When tensor A is evicted and is needed to recompute tensor B, B's cost increases.  We propagate this via `recomp_cnt` and `recomp_srcs`, matching Algorithms E/F from the paper.  This prevents the algorithm from evicting a chain of dependent tensors whose combined recomputation cost would be excessive.

---

## 8. Interpreting the Output

### Tensor Categorisation

```
  PARAM              count=20     total=   394.53 KB
  ACT                count=19     total=  4257.81 KB
  GRAD               count=0      total=     0.00 KB
  OPTIMIZER_STATE    count=60     total=   789.06 KB
  OTHER              count=766    total= 26267.68 KB
```

Five categories matching the course spec.  GRAD is 0 with `foreach=True` (see Decision 2).

### AC Selection Report

```
  Peak memory before AC:       506.10 MB
  Peak memory after AC:        431.25 MB
  Peak reduction:               74.85 MB
  Activation memory freed:     229.00 MB
  Extra computation cost:        1.05 ms
  Activation memory retained:  102.59 MB
```

The peak reduction is less than the activation memory freed because evicted tensors are still live during the forward pass — they are only freed during their idle period.

---

## 9. Known Limitations and Simplifications

**Swap branch omitted.**  The full mu-TWO algorithm considers both swap (CPU offloading) and recompute for each tensor, comparing overhead to choose the cheaper option.  We implement only the recompute branch, since the project focuses on activation checkpointing.  The profiler does measure swap times (for future extension), but the selection algorithm does not use them.

**No multiplexing.**  mu-TWO's Multiplexer (Algorithm C) overlaps swaps with computation from a second model's graph.  This is specific to multi-model training and not applicable to single-model AC.

**Static graph only.**  FX tracing captures a single trace.  Models with data-dependent control flow will produce different graphs for different inputs.

**Gradient identification with foreach=True.**  Gradient nodes cannot be distinguished from other backward-region tensors without the `_fused_adam` node.

**Phase 3 not yet implemented.**  The selection algorithm outputs two lists (recompute vs. retain) but does not yet modify the graph.  Phase 3 will use `_extract_graph_with_inputs_outputs()` to extract recomputation subgraphs and insert them before the first backward use.

---

## 10. Test Suite

The test suite lives in `tests/test_profiler.py` and contains **55 tests** runnable with:

```bash
python -m pytest tests/ -v
```

Tests that require CUDA are decorated with `@requires_cuda` and are skipped on CPU-only machines.

### 10.1 Testing philosophy

The key question: *what property does this code rely on that, if violated, would silently produce wrong profiling data or wrong AC decisions?*

For the profiler: separator nodes partition the graph correctly; every intermediate is a forward-region `call_function` with backward users; lifetime endpoints are consistent.

For the AC algorithm: every intermediate is accounted for (no overlap, full coverage); every recomputed node's inputs are available; the memory simulator is monotonic (more evictions never increase peak).

### 10.2 Test categories

| Class | Tests | What it validates |
|-------|------:|-------------------|
| `TestStaticAnalysis` | 13 | Separator detection, param extraction, intermediate identification |
| `TestNodeClassification` | 7 | PARAM/ACT/GRAD/OPTIMIZER_STATE/OTHER classification, region boundaries |
| `TestLifetimes` | 6 | `last_fwd < first_bwd`, correct regions, positive lifetime and size |
| `TestRuntimeProfiling` | 6 | Non-negative runtimes, populated stats, reset clears data |
| `TestACSelection` | 9 | Full coverage, no overlap, input availability, mem_limit behavior |
| `TestVisualizerOutput` | 5 | Timeline length, non-negative memory, role sums, PNG creation |
| `TestAPIGuardrails` | 5 | Missing separator raises, `_tensor_size_bytes` correctness |
| `TestArchitectures` | 4 | Single layer, deep network, larger dim, AC across sizes |

### 10.3 Running the tests

```bash
# Unit tests (CUDA tests skip if no GPU)
python -m pytest tests/test_profiler.py -v

# Full pipeline
python starter_code.py

# Benchmarks
python benchmarks.py Resnet18
python benchmarks.py Resnet50
python benchmarks.py Bert
python benchmarks.py Transformer

# Validation (profiler accuracy + AC sanity + simulator consistency)
python validate.py --all
```

Expected: 55 unit tests pass on CUDA (4 on CPU).  Validation passes on DummyModel, ResNet18, and BERT.
