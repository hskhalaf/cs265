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
   - 5.3 [Validation of Recompute Decisions](#53-validation-of-recompute-decisions)
6. [Design Decisions and Challenges](#6-design-decisions-and-challenges)
7. [Interpreting the Output](#7-interpreting-the-output)
8. [Known Limitations](#8-known-limitations)
9. [Test Suite](#9-test-suite)
   - 9.1 [Testing philosophy](#91-testing-philosophy)
   - 9.2 [Test categories](#92-test-categories)
   - 9.3 [Running the tests](#93-running-the-tests)

---

## 1. Introduction

Training a deep neural network requires far more GPU memory than the model weights alone.  During a single training iteration the GPU must simultaneously hold parameters, activations (intermediate outputs of each layer), gradients, and optimizer state (Adam's two moment buffers per parameter).  Activations account for 70–85% of peak memory because they are produced in the forward pass but cannot be freed until their corresponding backward computation consumes them — and the two passes run in opposite order, so the earliest activations live the longest.

**Activation checkpointing** addresses this by discarding selected activations after the forward pass and recomputing them from retained values during the backward pass, trading extra computation for a lower memory peak.  This project implements the infrastructure required to make that trade-off intelligently:

- **Phase 1 (Graph Profiler)** traces one full training step into a single FX graph, executes it node-by-node to collect per-operator timing and memory statistics, classifies every tensor by role (parameter, activation, gradient, other), computes activation lifetimes, and produces a peak memory breakdown chart.

- **Phase 2 (AC Selection Algorithm)** uses the profiler's output to implement the mu-TWO greedy algorithm, which ranks each activation by its *recompute ratio* (bytes freed per millisecond of extra computation) and greedily selects which to discard until peak memory fits within a budget.

Phase 3 (graph rewriting to insert recomputation nodes) will be implemented next.

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

Everything before `%sep` is the forward pass.  Everything between `%sep` and `%sep_backward` is the loss computation.  Everything after `%sep_backward` is the backward pass.  The optimizer step (`_fused_adam`) follows at the end.

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
activation_checkpoint.py  Phase 2.  mu-TWO greedy selection algorithm.
visualizer.py           Memory breakdown stacked bar chart.
starter_code.py         Entry point: profile → visualise → select → (rewrite).
benchmarks.py           Provided by course.  ResNet / Transformer benchmarks.
utils.py                Provided by course.  SPMD decomposition table.
tests/test_profiler.py  54-test suite across 8 categories.
```

The execution flow is:

```
starter_code.experiment()
  ├── Build DummyModel, Adam optimizer
  ├── Initialise optimizer state (Adam lazy init)
  └── compile(train_step, graph_transformation)
        ├── graph_tracer._compile() → GraphModule
        └── graph_transformation(gm, args)
              ├── GraphProfiler(gm)           ← static analysis
              ├── profiler.run(*args) × N     ← warm-up
              ├── profiler.reset_stats()
              ├── profiler.run(*args) × M     ← measurement
              ├── profiler.aggregate_stats()
              ├── profiler.print_stats()      ← report
              ├── MemoryVisualizer(profiler)
              │     └── plot_memory_timeline() ← PNG chart
              ├── select_activations_to_recompute(profiler)
              └── print_ac_decisions()        ← selection report
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

These two indices partition the graph into four regions: FORWARD (index <= sep), LOSS (between sep and sep_backward), BACKWARD (>= sep_backward), and OPTIMIZER (>= fused_adam).

#### Step 2 — Extract parameters and gradients

The `_fused_adam` node's arguments are structured lists:

```python
self.param_nodes = set(self.optimizer_node.args[0])  # parameter tensors
self.grad_nodes  = set(self.optimizer_node.args[1])  # gradient tensors
```

This is how we identify which placeholder nodes are model parameters versus optimizer states versus batch data: parameters are exactly those that appear in `args[0]` of the Adam node.

#### Step 3 — Identify intermediate activations

An intermediate activation is defined as a node that satisfies all four conditions:

1. It is a `call_function` node (a computation, not an input or output).
2. Its index is strictly before `sep_index` (produced during the forward pass).
3. It is not a parameter node.
4. It has at least one user whose index is at or after `sep_bwd_index` (consumed during the backward pass).

These are exactly the tensors that activation checkpointing can target: they are produced in the forward pass, sit idle in GPU memory while the forward pass completes and the backward pass reaches them, and are then consumed by a backward operation.

#### Step 4 — Compute lifetimes

For each intermediate we record two indices:

- **`last_fwd_access`** — the maximum index among users that fall within the forward region.  This is the last step at which the activation is actively used before becoming idle.
- **`first_bwd_access`** — the minimum index among users that fall within the backward region.  This is the first step at which the backward pass needs this activation.

The difference `first_bwd_access - last_fwd_access` is the **lifetime** — the number of steps during which the tensor sits idle in GPU memory.  Longer lifetimes mean more memory wasted, making the activation a better candidate for checkpointing.

#### Step 5 — Compute tensor sizes

Each node's output shape and dtype are available from the FakeTensor in `node.meta['val']`, recorded during symbolic tracing:

```python
def _tensor_size_bytes(val):
    if isinstance(val, torch.Tensor):
        return val.numel() * val.element_size()
    ...
```

This gives us the memory footprint of each activation without executing any real operations.

### 4.2 Runtime Profiling

The `run_node` override does three things for every node:

#### Timing

We bracket each node execution with `torch.cuda.Event` pairs:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
result = super().run_node(n)
end_event.record()
torch.cuda.synchronize()

elapsed_ms = start_event.elapsed_time(end_event)
```

`torch.cuda.synchronize()` blocks until all GPU kernels complete, ensuring the timing captures actual kernel execution rather than just Python-side dispatch.

#### Memory tracking

Before and after each node, we read `torch.cuda.memory_allocated()`.  The delta tells us how much GPU memory the operation allocated (net).  Positive deltas indicate new allocations; negative deltas indicate that the operation freed more memory than it allocated (e.g. an in-place update that releases a temporary).

#### Activation swapping

During profiling, intermediate activations are swapped to and from CPU memory to (a) measure the actual swap overhead per tensor and (b) prevent GPU OOM when profiling large models.  The schedule is precomputed from the lifetime analysis:

- **Swap-out**: at the step corresponding to `last_fwd_access` of intermediate `x`, move `x` to CPU with `.cpu()` and store it in `self._cpu_store`.
- **Swap-in**: at the step corresponding to `first_bwd_access` of intermediate `x`, move it back with `.cuda()` and write it into the interpreter's environment (`self.env[source_node]`).

Both transfers are timed with CUDA events.  The measured `swap_out_time_ms` and `swap_in_time_ms` are stored in the `IntermediateInfo` dataclass and used by the selection algorithm.

#### Aggregation

The profiler is designed to be run for `W` warm-up iterations followed by `M` measurement iterations.  Between the two phases, `reset_stats()` clears all accumulators.  After the measurement phase, `aggregate_stats()` averages all per-node measurements:

```python
self.avg_runtimes[name] = sum(runs) / len(runs)
```

It also populates the `IntermediateInfo` fields (`recompute_cost_ms`, `inactive_time_ms`, `swap_out_time_ms`, `swap_in_time_ms`) that Phase 2 consumes.  The `inactive_time_ms` is approximated as the sum of node runtimes over the interval `(last_fwd_access, first_bwd_access)` — the wall-clock time the activation sits unused.

### 4.3 Memory Visualisation

`visualizer.py` produces a stacked bar chart showing live memory at each step in the timeline, broken down by tensor role:

- **Blue** = PARAM — parameters are live for the entire iteration.
- **Green** = ACT — activations accumulate during forward, freed during backward.
- **Orange** = GRAD — gradients grow during backward, consumed by optimizer.
- **Purple** = OTHER — batch data, loss, optimizer states.

A tensor with size `s` is counted as live at step `t` if `produced_at <= t <= last_use`.  The peak step is marked with a dashed vertical line.  Separator boundaries (`%sep`, `%sep_backward`) are marked with dotted vertical lines.

The chart answers the key question: *what dominates memory at the peak?*  If green (activations) dominates, activation checkpointing will help.  If it's parameters or optimizer state, you need a different strategy.

---

## 5. Phase 2: Activation Checkpointing Selection

### 5.1 The Problem

Given `N` intermediate activations, each with a known memory size `mem_i` and recomputation cost `cost_i`, choose which to **retain** in GPU memory and which to **discard** (recompute during backward).  Discarding an activation frees `mem_i` bytes from GPU memory but costs `cost_i` milliseconds of extra computation.  The goal is to reduce peak memory below a budget while minimising the extra computation.

Choosing the optimal subset is NP-hard in general (it reduces to a variant of the knapsack problem).  We use the greedy heuristic from mu-TWO.

### 5.2 The mu-TWO Greedy Algorithm

mu-TWO (Purandare et al., MLSys 2023) is a multi-model training compiler.  Its activation checkpointing component uses a greedy iterative approach.  For single-model training, the relevant ranking metric is the **recompute ratio**:

```
recompute_ratio_i = mem_i / cost_i
```

This measures how many bytes of GPU memory we free per millisecond of extra computation.  A high ratio means the activation is cheap to recompute relative to the memory it consumes — the best "bang for buck."

The algorithm is:

```
1.  For each intermediate activation, compute recompute_ratio.
2.  Sort by recompute_ratio descending.
3.  Greedily mark the top-ranked activation for recomputation.
4.  Accumulate the memory freed.
5.  Stop when accumulated savings >= memory_budget (or all processed).
6.  Validate the recompute set (see 5.3).
```

In code:

```python
candidates.sort(key=lambda c: c["recompute_ratio"], reverse=True)

for cand in candidates:
    if memory_budget is not None and memory_saved >= memory_budget:
        to_retain.append(cand["node"])
        continue
    to_recompute.append(cand["node"])
    memory_saved += cand["memory_size"]
```

When no budget is specified, the algorithm defaults to recomputing every valid intermediate (maximum memory savings).

### 5.3 Validation of Recompute Decisions

Not every activation can be recomputed.  To recompute activation `x` during the backward pass, every input to the operation that produced `x` must be available at that point.  An input is available if it is either:

(a) a **placeholder** node — parameters, optimizer states, and batch data are always in GPU memory, or
(b) a **retained** intermediate — an activation we chose to keep.

If an activation's inputs include another activation that was also marked for recomputation, neither can be recomputed (circular dependency).  The validation pass iterates until stable:

```python
while changed:
    changed = False
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

Moving one node from recompute to retain may make another node's inputs valid (since retained nodes are available), so we iterate.

---

## 6. Design Decisions and Challenges

### Decision 1: FX-based profiling instead of hooks

The Milestone 1 prototype used PyTorch module hooks to instrument execution.  Hooks observe the model from the outside: they fire before/after each module's forward and backward, and we had to reconstruct the graph, track tensor identity across callbacks, and infer implicit autograd saves through a 4-pass static analysis.

FX tracing eliminates all of these problems.  The traced graph is a single, explicit DAG with stable node names, exact data dependencies (via `node.users`), and both forward and backward operations in one structure.  Lifetimes are directly readable from the graph — no inference required.  And critically, the graph is rewritable, which Phase 3 requires.

The trade-off is that FX tracing captures a single static graph (no data-dependent control flow) and requires the model to be traceable by `make_fx()`.  For the models in this project (DummyModel, ResNet, Transformer), this is not a limitation.

### Decision 2: Swapping intermediates during profiling

The `run_node` method swaps intermediates to CPU after their last forward use and back to GPU before their first backward use.  This serves two purposes:

1. **Measurement** — we record actual swap latencies that mu-TWO's full algorithm uses for the swap-vs-recompute decision.
2. **Memory management** — for large models (ResNet-152, BERT), keeping all intermediates in GPU memory during profiling may cause OOM.  Swapping ensures the profiler can run on models that barely fit in memory.

The downside is that profiling itself adds overhead (the swap transfers).  This overhead is not included in the per-node timing measurements, since swaps happen outside the `super().run_node()` call.

### Decision 3: Recompute ratio as the selection metric

mu-TWO's full algorithm considers both swapping (offloading to CPU) and recomputation, choosing whichever has lower overhead for each tensor.  Since this project focuses on activation checkpointing (not CPU offloading), we use only the recompute branch:

```
recompute_ratio = memory_size / recompute_cost
```

Tensors with high ratios are cheap to recompute relative to their memory footprint.  This is a strictly greedy heuristic — it does not consider interactions between choices (e.g. two activations that share an input and could be recomputed together for less cost than the sum of their individual costs).  For sequential models like our DummyModel, where each activation depends only on the previous one, the greedy approach produces optimal or near-optimal results.

### Challenge 1: Identifying the optimizer node

The `_fused_adam` node is not always present.  If the optimizer is SGD or the decomposition table is different, the node target changes.  Currently we hard-code `torch.ops.aten._fused_adam.default`.  A more robust approach would search for any optimizer-related node by inspecting the OPTIMIZER region.

### Challenge 2: FakeTensor metadata for size computation

`node.meta['val']` contains a FakeTensor only for `call_function` nodes traced through `make_fx` in fake mode.  Placeholder nodes (parameters, batch data) do not have FakeTensor metadata unless explicitly annotated.  We handle this by returning 0 for nodes without metadata and computing parameter sizes separately from `self.param_nodes`.

### Challenge 3: Validation iteration for recompute set

The validation pass must iterate until convergence because moving one node from recompute to retain changes the set of available inputs, potentially making a previously invalid choice valid.  In the worst case this is O(N^2) in the number of intermediates, but N is small (tens to hundreds for typical models) and the loop rarely iterates more than twice.

---

## 7. Interpreting the Output

### Operation Summary Table

```
#    Node                                Region     Time(ms)       Mem(B)
0    l__self___mod_0                      FORWARD       0.045         +512
1    relu                                 FORWARD       0.012         +512
...
42   t_3                                  BACKWARD      0.031         -512
...
58   _fused_adam                           OPTIMIZER     0.250        +1024
```

- **#** — Index in topological order.
- **Node** — FX node name (derived from the ATen operator or module name).
- **Region** — FORWARD, LOSS, BACKWARD, or OPTIMIZER.
- **Time(ms)** — Average wall-clock time for this operation (from CUDA event timing).
- **Mem(B)** — Average net GPU memory change (positive = allocation, negative = deallocation).

### Tensor Categorisation

```
  PARAM              count=8      total=    50.00 KB
  ACT                count=12     total=   375.00 KB
  GRAD               count=8      total=    50.00 KB
  OTHER              count=45     total=   100.00 KB
```

Counts and total memory per tensor role.  ACT count should equal the number of identified intermediate activations.  PARAM count should match `len(list(model.parameters()))`.

### Intermediate Activation Lifetimes

```
Name                         Size(KB)   LastFwd  FirstBwd  Lifetime  Recomp(ms)
relu                            25.00        3        38        35      0.0120
relu_1                          25.00        7        34        27      0.0115
...
```

- **LastFwd** — Index of the last forward-region step that uses this activation.
- **FirstBwd** — Index of the first backward-region step that needs it.
- **Lifetime** — `FirstBwd - LastFwd`.  The number of steps the activation sits idle.
- **Recomp(ms)** — Time to recompute this activation (averaged over measurement runs).

**Key insight:** activations with large `Size × Lifetime` products contribute most to peak memory.  Those with high `Size / Recomp` ratios are the best candidates for checkpointing.

### AC Selection Report

```
  Intermediates to RECOMPUTE (8):
    relu                     25.00 KB      0.0120 ms     ratio=2133333
    relu_1                   25.00 KB      0.0115 ms     ratio=2228260
    ...

  Intermediates to RETAIN (4):
    mm_2                     12.50 KB

  Summary:
    Memory saved: 200.00 KB
    Extra computation: 0.0960 ms
    Memory retained: 50.00 KB
```

### Memory Breakdown Chart

The PNG chart shows a stacked bar at each step.  Look for:

- The peak (dashed line) and its composition.  If green (ACT) dominates, checkpointing is the right strategy.
- The forward/backward boundary (dotted lines).  Activations should accumulate through the forward pass and decay through the backward pass.

---

## 8. Known Limitations

**Static graph only.**  FX tracing captures a single trace.  Models with data-dependent control flow (`if`, dynamic loops) will produce different graphs for different inputs.  The profiler captures one trace and assumes it is representative.

**Operator granularity.**  The profiler operates at the ATen operator level (individual matmuls, relu calls), which is finer than the module level but may still merge operations that the hardware executes as fused kernels.

**_fused_adam assumption.**  The optimizer identification is hard-coded to `_fused_adam`.  Other optimizers (SGD, AdamW, RMSprop) would require additional target checks.

**No multi-iteration analysis.**  The profiler captures a single training iteration.  Learning rate schedules, gradient accumulation, and warmup effects are not modelled.

**Greedy AC selection is approximate.**  The mu-TWO greedy algorithm does not guarantee optimal memory reduction for a given compute budget.  It does not consider subgraph sharing (two activations that can be recomputed together) or memory fragmentation effects.

**Phase 3 not yet implemented.**  The selection algorithm outputs two lists (recompute vs. retain) but does not yet modify the graph.  Phase 3 will use `_extract_graph_with_inputs_outputs()` to extract recomputation subgraphs and insert them before the first backward use.

---

## 9. Test Suite

The test suite lives in `tests/test_profiler.py` and contains **54 tests** runnable with:

```bash
python -m pytest tests/ -v
```

Tests that require CUDA are decorated with `@requires_cuda` and are skipped on CPU-only machines.  Four tests (utility functions, error handling) run on CPU.

### 9.1 Testing philosophy

The key question for each component: *what property does this code rely on that, if violated, would silently produce wrong profiling data or wrong AC decisions?*

For the profiler, the critical invariants are:
- Separator nodes are found and correctly partition the graph.
- Every intermediate is a forward-region `call_function` with backward users.
- Lifetime endpoints are consistent: `last_fwd < first_bwd`, both within their respective regions.
- Runtime measurements are non-negative and populated after aggregation.

For the AC algorithm:
- Every intermediate is accounted for (recompute + retain = all intermediates, no overlap).
- Every recomputed node's inputs are available (placeholders or retained).
- The greedy order matches the ranking by recompute ratio.

### 9.2 Test categories

| Class | Tests | What it validates |
|-------|------:|-------------------|
| `TestStaticAnalysis` | 13 | Separator detection, param/grad extraction, intermediate identification, node indexing |
| `TestNodeClassification` | 7 | PARAM/ACT/GRAD/OTHER classification, region assignment, region boundary consistency |
| `TestLifetimes` | 6 | `last_fwd < first_bwd`, both in correct regions, positive lifetime, positive memory size |
| `TestRuntimeProfiling` | 6 | Non-negative runtimes, populated stats, reset clears data, recompute cost > 0 |
| `TestACSelection` | 8 | Full coverage (recompute + retain = all), no overlap, input availability, budget=0, greedy order |
| `TestVisualizerOutput` | 5 | Timeline length, non-negative memory, role sums equal total, PNG file created |
| `TestAPIGuardrails` | 5 | Missing separator raises, `_tensor_size_bytes` correctness, multiple model sizes |
| `TestArchitectures` | 4 | Single layer, deep network, larger dim, AC selection across all sizes |

### 9.3 Running the tests

```bash
# All tests (CUDA tests skip if no GPU)
python -m pytest tests/test_profiler.py -v

# CPU-only tests
python -m pytest tests/test_profiler.py -v -k "not cuda"

# Full pipeline (requires CUDA)
python starter_code.py
```

Expected output on CUDA: 54 passed.  On CPU: 4 passed, 50 skipped.
