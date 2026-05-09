# Phase 1 — Graph Profiling

A self-contained walk-through of what we built, why we built it that way,
and the surprises we hit along the way.

---

## 1. The goal

Training a neural network on a GPU spends most of its memory on things
you do not directly think about: forward activations that have to stay
alive until the backward pass uses them, gradient buffers, optimizer
state.  If you want to *reduce* that memory (Phase 2 / 3 do this with
activation checkpointing) you first have to *see* it.

Phase 1 is that "seeing" tool — a static profiler that, given one model
and batch size, produces:

1. **A stacked-area chart** of live GPU memory over the course of one
   training iteration, broken down by tensor role
   (parameters / activations / gradients / other).  One plot per
   `(model, batch_size)`.
2. **A bar chart** of peak memory across batch sizes, one per model.

Plus per-iteration latency, per-node timing, and the list of
intermediate activations and their lifetimes — everything Phase 2 will
need.

---

## 2. Setup: tracing the training step

A training iteration is four phases glued end-to-end:

```
[ FORWARD ]  →  [ LOSS ]  →  [ BACKWARD ]  →  [ OPTIMIZER STEP ]
```

We use PyTorch FX (`make_fx` via `graph_tracer.compile`) to trace one
invocation of `train_step(model, optim, inputs)` into a flat
`fx.GraphModule` containing every primitive op (`matmul`, `convolution`,
`add`, `_foreach_add_`, etc.) PyTorch would execute.  Two custom
"separator" ops, `sep` and `sep_backward`, are inserted by `SEPFunction`
in `graph_tracer.py` so the profiler can locate the forward/loss and
backward/optimizer boundaries inside the flat op list.

The result is a graph of N nodes (N ranges from ~800 for the dummy MLP
to ~8800 for BERT-base).  Each node has:

- an `op` (placeholder, call_function, output)
- a `target` (the ATen op or builtin)
- `args` and `all_input_nodes` (its inputs)
- `users` (its consumers)
- `meta["val"]` — a `FakeTensor` describing the output's shape and dtype

That metadata is everything we need.  We never execute the graph to
compute the memory chart; the chart comes from analyzing this object.

---

## 3. The profiler

`GraphProfiler` (in `graph_prof.py`) subclasses `torch.fx.Interpreter`.
The constructor runs six static-analysis passes (each depends on the
previous), then accepts CUDA-event timings via `run()` / `run_node()`
during the measurement loop in `starter_code.py`.

### 3.1 Six static passes

| Pass | What it does |
| --- | --- |
| `_find_separators` | locate `sep_idx`, `sep_bwd_idx`, `opt_idx` in the node list |
| `_assign_regions` | label every node FORWARD / LOSS / BACKWARD / OPTIMIZER |
| `_classify_tensors` | label every node PARAM / GRAD / OTHER (ACT is filled in next) |
| `_find_intermediates` | identify forward call_functions whose output is read in backward → these become ACT |
| `_compute_sizes` | per-node logical bytes and per-node *allocation* bytes |
| `_compute_aliases` | map each node to its storage *owner* (see §4.2 — the central trick) |

After these run the profiler knows the static structure of the
iteration without having executed a single op.

### 3.2 What we compute is exact, not estimated

For every node, all three pieces of data come straight from the FX
graph — no approximation:

- **Output bytes**: `meta["val"]` carries the exact shape and dtype the
  op produces.  `numel × element_size` is the exact byte count the GPU
  allocator will use.
- **Lifetime**: `node.users` plus topological order tells us exactly
  when a tensor is first produced and last consumed.  The interval
  `[produced, last_use]` is a deterministic property of the graph.
- **Aliasing**: the ATen schema's `alias_info` tells us exactly which
  ops mutate inputs vs. allocate fresh storage.

So at every step, "sum the bytes of all owners whose lifetime includes
this step" is an exact count of the bytes held by FX-graph-visible
tensors.

The peak from this static walk and the peak the GPU actually sees
during execution don't match perfectly — the GPU number is larger by
the size of memory that exists *because the model runs*, not because of
any property of the graph: cuDNN/cuBLAS workspaces a kernel grabs
internally for an algorithm, op-internal scratch buffers freed before
the kernel returns.  These are real `cudaMalloc` calls but they aren't
the output of any FX node, so the static analysis can't see them.  We
acknowledge this and don't try to attribute it.

### 3.3 The live-memory timeline

`memory_timeline_by_role()` is the heart of the chart:

```
for each storage owner:
    for each step t in [produced, last_use]:
        timeline[role][t] += owner.bytes
```

Then `peak_memory_bytes()` is just the max sum across steps.  The chart
plots the per-role timelines as a stacked area; the dashed vertical
line at the peak step shows where the maximum is.

### 3.4 Per-node timing

For Phase 2's recompute-cost estimate we need per-node runtime.  Two
warm-up iterations stabilize cuDNN's algorithm picker and the caching
allocator; three measurement iterations are averaged.  Inside each
`run_node`:

```python
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
result = super().run_node(n)
end.record()
torch.cuda.synchronize()
self._runtimes_ms[n.name].append(start.elapsed_time(end))
```

The whole-iteration latency is timed the same way at the `run()` level.

---

## 4. The three real challenges

Each one bit us during development; documenting them so the choices in
the code are explicable.

### 4.1 Placeholder lifetime

**Symptom**: on ResNet18, the static walk reported 0.04 MB live at step
0 and 42 MB at the last step — nowhere near the ~143 MB the model + Adam
moments actually take.

**Cause**: a placeholder is one specific node at index `i` in the FX
graph (e.g., `arg0_42` is "the 42nd parameter of the model").  The
naive walk says it's live from `idx[arg0_42]` to `max(idx of users)`.
But all placeholders are passed in *simultaneously* at the start of
`run()`, and they're held by `model` and `optim` Python objects *across*
iterations.  Their physical lifetime is **the entire iteration**, not
just from their node index to the optimizer step that last touches them.

**Fix**: in `memory_timeline_by_role`, special-case placeholders to be
live from `lo=0` to `hi=N-1`.  After the fix, ResNet18's chart shows a
flat ~143 MB baseline (PARAM + opt-state) running across the whole
iteration, with the activation hump on top.

### 4.2 Aliasing — the central abstraction

This is the single most important idea in the profiler.  Three FX
patterns make several distinct nodes share one underlying allocation,
and counting each node naively double-counts its bytes (or worse —
N-times-counts on a 62-element foreach).

#### Pattern A — schema aliasing (in-place / view ops)

Examples: `relu_`, `view`, `transpose`, `_foreach_add_`,
`_fused_adam`.  Their ATen schema marks return tensors as
`alias_info != None`.  Detected with `_alias_in_schema`: allocation
contribution = 0; the storage chain points back to the input.

#### Pattern B — getitem unpacking of a multi-output op

`_foreach_div(a_list, b_list)` returns `List[Tensor]` of 62 elements on
ResNet18; each element is then accessed by an `operator.getitem` node.
There are two flavors and they need different handling:

- **Aliasing parent** (e.g. `aten.split` whose outputs view the input;
  `_foreach_mul_` whose outputs alias the inputs in-place): each
  `getitem(parent, i)` should chain through to the *i-th* input
  element.  `_alias_target` does this — for single-input ops (split)
  all getitems point to that input; for list-input ops (foreach)
  `getitem(parent, i)` points to `parent.args[0][i]`.
- **Independent parent** (e.g. `convolution_backward` returns three
  *new* tensors): each getitem owns *its own* element bytes, so the
  per-element role can be different (one element is GRAD, another is
  OTHER).  We had to fight specifically for this — see §4.3.

#### Pattern C — placeholder persistence

Already discussed in §4.1.  Implementation-wise it's just a branch in
`memory_timeline_by_role` that sets `lo=0`, `hi=N-1` for placeholders.

#### The unifying mechanism

`_compute_aliases` builds a single `storage_owner: Dict[Node, Node]`
map.  For every node it asks "who really owns my bytes?" and writes the
answer.  Then `memory_timeline_by_role` iterates by owner (each owner
contributes its size exactly once across the right interval), and
merges roles across all aliases of an owner using `max()` over
`NodeType` (an `IntEnum` whose values double as priority:
`PARAM > GRAD > ACT > OTHER`).

### 4.3 Per-element role attribution (the `conv_backward` trap)

This was the bug that ate the most time.  `conv_backward` returns
three independent tensors:

```
(grad_input, grad_weight, grad_bias) = convolution_backward(...)
```

`grad_input` is an *activation* gradient (it flows into the upstream
conv's backward op and is then freed).  `grad_weight` and `grad_bias`
are *parameter* gradients (they flow into the optimizer's foreach call).
By role they should split: grad_input → OTHER, grad_weight/bias → GRAD.

A first attempt at "always make getitem an alias of its parent" lumped
all three under the parent's owner.  When we then merged roles across
the parent's aliases, the GRAD-classified getitems forced the *whole*
parent's bytes (including the much larger grad_input) into the GRAD
bucket.  On ResNet18 bs=32, GRAD jumped from a correct ~44 MB to a
wrong 593 MB.

The right policy (in `_allocation_bytes`):

- if the parent has `alias_info` on its returns → it's a view-/in-place
  multi-output → getitem aliases the parent's input;
- otherwise (parent allocates independent tensors) → each getitem owns
  its own element bytes and gets its own role.

After the fix, ResNet18 bs=32 GRAD is back at ~44 MB, the activation
gradients live in OTHER where they belong, and the BERT optimizer chain
(`denom = _foreach_sqrt(v); _foreach_mul_(denom, ...);
_foreach_addcdiv_(...)`) correctly counts the scratch buffer once.

---

## 5. Results

Running `python starter_code.py -b 4 8 16 32` produces 16 breakdown
charts (4 models × 4 batch sizes) plus 4 peak-vs-batch charts (one per
model).  Headline numbers from a full sweep (peak = static FX walk):

| Model    |  BS | Peak (MB) | Latency (ms) |
|----------|----:|----------:|-------------:|
| dummy    |   4 |       1.9 |        282.9 |
| dummy    |   8 |       2.3 |        294.8 |
| dummy    |  16 |       2.9 |        310.2 |
| dummy    |  32 |       4.1 |        281.1 |
| resnet18 |   4 |     258.3 |        910.5 |
| resnet18 |   8 |     331.5 |        961.4 |
| resnet18 |  16 |     498.4 |        986.6 |
| resnet18 |  32 |     832.2 |       1046.6 |
| resnet50 |   4 |     653.0 |       2414.1 |
| resnet50 |   8 |     972.3 |       2597.2 |
| resnet50 |  16 |    1612.1 |       2630.3 |
| resnet50 |  32 |    2915.7 |       2795.0 |
| bert     |   4 |    2088.3 |       3447.7 |
| bert     |   8 |    2088.3 |       3528.2 |
| bert     |  16 |    2505.9 |       3792.7 |
| bert     |  32 |    3739.3 |       3913.3 |

**Reading the breakdown plots**

For ResNet (the canonical case):
- The **PARAM** band at the bottom is constant — model parameters
  persist all iteration.
- The **green ACT hump** rises during forward (each layer's output
  must stay alive until backward consumes it), peaks just before the
  forward/backward boundary, and falls during backward as activations
  are released.
- The **orange GRAD band** appears during backward as parameter
  gradients are produced.
- **OTHER** (gray) is the constant baseline of optimizer state (Adam
  m + v) plus per-step optimizer scratch buffers in the optimizer
  region.

For BERT, peak shifts: at small batch the peak is in the optimizer
region (Adam moments dominate); at large batch the activation hump in
the forward/backward region exceeds it.

For dummy MLP, the model is too small for activations to dominate; the
chart is mostly the constant parameter and optimizer-state baseline.
The bar chart is essentially flat — useful as a profiler-correctness
check, less useful for showing batch-size scaling.

**Reading the peak-vs-batch plots**

ResNet18 / ResNet50 / BERT all scale roughly linearly with batch size
once the activation hump dominates.  Dummy doesn't scale because it has
no meaningful activation memory at this size.

---

## 6. Known limitations

- **Workspace memory is invisible.** cuDNN/cuBLAS scratch and other
  op-internal allocations are real GPU bytes but aren't outputs of FX
  nodes, so the static walk doesn't include them.  Typically 10–50 MB
  on ResNet, ~9 MB on dummy — small relative to the activation hump.
- **Static walk can over-extend lifetime in foreach in-place chains**
  if the order of `parent.args[0]` doesn't match the output ordering —
  we trust that `output[i]` aliases `args[0][i]`, which is true for the
  ATen `_foreach_*_` family but not formally guaranteed for every op
  with `alias_info`.
- **Multi-output ops with mixed users** (some users are getitems, some
  are not) keep the parent's full bytes; this case is rare in the
  models we tested but isn't perfectly handled.

---

## 7. Files

| File | Purpose |
|---|---|
| [starter_code.py](starter_code.py) | Phase 1 entry point — sweep `(model, batch_size)`, save plots |
| [graph_prof.py](graph_prof.py) | the `GraphProfiler` and the static analysis described above |
| [models.py](models.py) | factory: dummy MLP, ResNet18/50, BERT-base + Adam(foreach=True) |
| [visualizer.py](visualizer.py) | the two plot helpers |
| [graph_tracer.py](graph_tracer.py) | provided — `compile()` + `SEPFunction` |
| [utils.py](utils.py) | provided — `SPMD_DECOMP_TABLE` decomposition rules |

To run:

```bash
python starter_code.py                        # everything
python starter_code.py resnet18 -b 8 16 32    # one model, three batch sizes
```
