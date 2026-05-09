# Phase 1 — Graph Profiling

A self-contained walk-through of what we built, why we built it that way,
what surprises we hit, and how we know the result is right.

---

## 1. The goal

Training a neural network on a GPU spends most of its memory on things you
do not directly think about: forward activations that have to stay alive
until the backward pass uses them, gradient buffers, optimizer state.  If
you want to *reduce* that memory (Phase 2 / 3 do this with activation
checkpointing) you first have to *see* it.

Phase 1 builds that "seeing" tool — a profiler that, given one model and
batch size, produces:

1. **A stacked-area chart** of live GPU memory over the course of one
   training iteration, broken down by tensor role
   (parameters / activations / gradients / other).  One plot per
   `(model, batch_size)`.
2. **A bar chart** of peak memory across batch sizes, one per model.

Plus per-iteration latency, per-node timing, the list of intermediate
activations and their lifetimes — everything Phase 2 will need.

---

## 2. Setup: tracing the training step

A training iteration is four phases glued end-to-end:

```
[ FORWARD ]  →  [ LOSS ]  →  [ BACKWARD ]  →  [ OPTIMIZER STEP ]
```

We use PyTorch FX (`make_fx` via `graph_tracer.compile`) to trace one
invocation of `train_step(model, optim, inputs)` into a flat
`fx.GraphModule` containing every primitive op (`matmul`, `convolution`,
`add`, `_foreach_add_`, etc.) that PyTorch would execute.  Two custom
"separator" ops, `sep` and `sep_backward`, are inserted by `SEPFunction`
([graph_tracer.py:46-56](graph_tracer.py#L46-L56)) so the profiler can
locate the forward/loss and backward/optimizer boundaries inside the flat
op list.

The result is a graph of N nodes (N ranges from ~800 for the dummy MLP
to ~8800 for BERT-base).  Each node has:

- an `op` (placeholder, call_function, output)
- a `target` (the ATen op or builtin)
- `args` and `all_input_nodes` (its inputs)
- `users` (its consumers)
- `meta["val"]` — a `FakeTensor` describing the output's shape and dtype

That metadata is everything we need for the static analysis.

---

## 3. The profiler

`GraphProfiler` ([graph_prof.py](graph_prof.py)) subclasses
`torch.fx.Interpreter`.  Its constructor runs six static-analysis passes
(in order, each depends on the previous), then accepts runtime
measurements via `run()` / `run_node()`.

### 3.1 Five static passes

| Pass | What it does |
| --- | --- |
| `_find_separators` | locate `sep_idx`, `sep_bwd_idx`, `opt_idx` in the node list |
| `_assign_regions` | label every node FORWARD / LOSS / BACKWARD / OPTIMIZER |
| `_classify_tensors` | label every node PARAM / GRAD / OTHER (ACT is filled in next) |
| `_find_intermediates` | identify forward call_functions whose output is read in backward → these become ACT |
| `_compute_sizes` | per-node logical size and per-node *allocation* size |
| `_compute_aliases` | map each node to its storage *owner* (see §4.2 — the central trick) |

After these run, the profiler knows the static structure of the iteration
without having executed a single op.

### 3.2 Runtime measurements

Two warm-up iterations (so cuDNN's algorithm picker and the caching
allocator stabilize), then three measurement iterations.  Inside each
`run_node`, before running the op:

```python
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
```

After the op:

```python
torch.cuda.synchronize()
self._runtimes_ms[n.name].append(start.elapsed_time(end))
self._cur_mem.append(torch.cuda.memory_allocated())
self._cur_peak.append(torch.cuda.max_memory_allocated())
```

`memory_allocated()` is the steady-state live-tensor total at the moment
of the snapshot (after the node produced its output, before the
interpreter's GC runs); `max_memory_allocated()` since the per-node reset
captures the peak *during* the op call, including transient cuDNN/cuBLAS
workspaces that the op allocates and frees within a single kernel
launch.

The three measurement runs are averaged in `aggregate_stats()`.

### 3.3 The live-memory timeline

`memory_timeline_by_role(include_runtime_residual=True)` is the heart of
the chart.  Conceptually:

```
for each storage owner:
    for each step t in [produced, last_use]:
        timeline[role][t] += owner.bytes
```

Then, if `include_runtime_residual=True`, the gap between this static
walk and the per-node measured peak is folded into OTHER step-by-step:

```
residual = measured[t] - sum(timeline[role][t] for role in NodeType)
if residual > 0:
    timeline[OTHER][t] += residual
```

So at every step the chart's **total** matches what the GPU actually
used; the per-role breakdown is the static walk's best attempt at
explaining where those bytes live.  When measured ≤ static (the static
walk slightly overshoots, e.g. on small-bs BERT in the optimizer
region), no residual is added.

---

## 4. The four real challenges

Each one bit us during development; documenting them here so the choices
in the code are explicable.

### 4.1 Placeholder lifetime

**Symptom**: on ResNet18, the static walk reported 0.04 MB live at step 0
while the GPU was holding 143 MB.  At the *last* step the static walk
reported 42 MB, but the GPU still had 143 MB.

**Cause**: a placeholder is one specific node at index `i` in the FX
graph (e.g., `arg0_42` is "the 42nd parameter of the model").  The naive
walk says it's live from `idx[arg0_42]` to `max(idx of users)`.  But all
placeholders are passed in *simultaneously* at the start of `run()`, and
they're held by `model` and `optim` Python objects *across* iterations.
Their physical lifetime is **the entire iteration**, not just from their
node index to the optimizer step that last touches them.

**Fix**: in `memory_timeline_by_role`, special-case placeholders to be
live from `lo=0` to `hi=N-1`.  After the fix, static at step 0 jumped
from 0 MB → 143 MB on ResNet18 (matches measured), and the chart no
longer "drops" at the end of the iteration.

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

### 4.4 Runtime residual — the cuDNN/cuBLAS workspace

Even after all the above, there's a small gap between the static walk
and the per-node measured peak — typically 10–50 MB on ResNet, ~9 MB
even on dummy.  This is *real* GPU memory that PyTorch's caching
allocator is holding for cuDNN / cuBLAS workspaces and op-internal
scratch buffers; it's not the output of any FX node, so the static walk
can't see it.

The `--no-cudnn` ablation we ran earlier showed that disabling cuDNN
only closed ~20 MB of a 300 MB ResNet50 gap, so this residual isn't
mainly cuDNN — it's a mix of small per-op things.  We don't try to
attribute them; we just fold the gap step-by-step into OTHER via
`include_runtime_residual=True`.  The result is a chart whose total
matches the GPU exactly.

---

## 5. How we validated this

`diagnose.py` is the primary debugging tool.  It runs one (model,
batch_size) and produces three things:

1. **Three peak numbers**: the static FX walk, the interpreter-measured
   peak, and an independent `gm()` invocation's `max_memory_allocated`.
   These should agree to within the runtime residual.
2. **Top-K largest discrepancy steps**: for each step, the gap
   `measured - static`, sorted by `|gap|`, with the op that ran at that
   step.  This is what localized §4.1, §4.3, and the BERT bug.
3. **Boundary check**: static vs measured at the start of forward,
   end of forward (sep), start of backward, start of optimizer, last
   step.  Tells us *which region* a discrepancy lives in.

Cross-checked against the deliverable bar chart, which uses
`torch.cuda.max_memory_allocated()` from a clean `gm()` run as the
gold-standard "what actually happened on the GPU" number — that
measurement is unaffected by anything in our static walk.

---

## 6. Results

Running `python starter_code.py -b 4 8 16 32` produces 16 breakdown
charts (4 models × 4 batch sizes) plus 4 peak-vs-batch charts (one per
model).  Headline numbers from one full sweep:

| Model    |  BS | Peak (MB) | Latency (ms) |
|----------|----:|----------:|-------------:|
| dummy    |   4 |      11.1 |        282.9 |
| dummy    |   8 |      12.3 |        294.8 |
| dummy    |  16 |      13.5 |        310.2 |
| dummy    |  32 |      11.1 |        281.1 |
| resnet18 |   4 |     275.6 |        910.5 |
| resnet18 |   8 |     379.4 |        961.4 |
| resnet18 |  16 |     548.2 |        986.6 |
| resnet18 |  32 |    1021.2 |       1046.6 |
| resnet50 |   4 |     671.5 |       2414.1 |
| resnet50 |   8 |    1009.2 |       2597.2 |
| resnet50 |  16 |    1686.1 |       2630.3 |
| resnet50 |  32 |    3059.7 |       2795.0 |
| bert     |   4 |    2108.6 |       3447.7 |
| bert     |   8 |    2120.0 |       3528.2 |
| bert     |  16 |    2526.8 |       3792.7 |
| bert     |  32 |    3753.5 |       3913.3 |

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
- After the optimizer step, only the constant PARAM + opt-state
  baseline remains (with the residual workspaces folded into OTHER).

For BERT, peak shifts: at small batch the peak is in the optimizer
region (Adam moments dominate); at large batch the activation hump in
the forward/backward region exceeds it.

For dummy MLP, the model is too small for activations to dominate; the
peak is always in the optimizer at ~11–13 MB and is mostly cuBLAS
workspace.  Useful to verify the profiler is correct, less useful to
show batch-size scaling.

**Reading the peak-vs-batch plots**

ResNet18 / ResNet50 / BERT all scale roughly linearly with batch size
once the activation hump dominates.  Dummy doesn't scale because
batch-dependent memory (~0.25 MB at bs=32) is dwarfed by the workspace
(~9 MB) which doesn't depend on batch.

---

## 7. Known limitations

- **Small over-count in BERT optimizer at small batch.** With my final
  alias logic this is now small (~10–20 MB) and the chart's residual-fold
  hides it visually.  At bs=4/8 BERT, peak is in the optimizer; at
  bs ≥ 16 the activation peak dominates and this isn't visible.
- **Static walk can over-extend lifetime in foreach in-place chains** if
  the order of `parent.args[0]` doesn't match the output ordering — we
  trust that `output[i]` aliases `args[0][i]`, which is true for the
  ATen `_foreach_*_` family but not formally guaranteed for every op
  with `alias_info`.
- **Workspace residual is not attributed to a kind**.  cuBLAS scratch
  for a matmul vs cuDNN scratch for a conv vs op-internal temporaries
  all flow into one OTHER bucket together; we can't separate them at
  the FX level.
- **Multi-output ops with mixed users** (some users are getitems, some
  are not) keep the parent's full bytes; this case is rare in the
  models we tested but isn't perfectly handled.

---

## 8. Files

| File | Purpose |
|---|---|
| [starter_code.py](starter_code.py) | Phase 1 entry point — sweep `(model, batch_size)`, save plots |
| [graph_prof.py](graph_prof.py) | the `GraphProfiler` and the static analysis described above |
| [models.py](models.py) | factory: dummy MLP, ResNet18/50, BERT-base + Adam(foreach=True) |
| [visualizer.py](visualizer.py) | the two plot helpers |
| [graph_tracer.py](graph_tracer.py) | provided — `compile()` + `SEPFunction` |
| [utils.py](utils.py) | provided — `SPMD_DECOMP_TABLE` decomposition rules |
| [diagnose.py](diagnose.py) | per-step `measured` vs `static` comparison; the validation tool |

To run:

```bash
python starter_code.py                        # everything
python starter_code.py resnet18 -b 8 16 32    # one model, three batch sizes
python diagnose.py bert -b 8                  # debug one configuration
```
