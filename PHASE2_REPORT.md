# Phase 2 — Activation Checkpointing

A self-contained walk-through of how Phase 2 turns the Phase 1 profiler
output into an activation-checkpointing decision: which intermediates to
*keep around* in GPU memory, and which to *throw away and recompute* in
the backward pass.

---

## 1. The goal

Phase 1 told us where the GPU memory goes; Phase 2 is the first
optimization that uses that information.  Activation checkpointing trades
**memory for compute**: any intermediate activation that's needed in the
backward pass can either be stored after the forward pass produces it (the
default — costs memory, costs no compute) or thrown away and rebuilt on
demand from the inputs that *are* alive at backward time (the AC choice —
saves memory, costs the recomputation work).

Phase 2 picks which activations to flip from "stored" to "recomputed."  The
deliverable, mirroring Phase 1, is two comparison plots per model:

1. **Peak memory vs batch size, no_ac vs ac** — grouped bars showing how
   far the µ-TWO greedy can shrink the peak.
2. **Latency vs batch size, no_ac vs ac** — the time cost AC pays for
   that memory savings.

Plus the per-`(model, batch)` breakdown chart, *for the AC-rewritten
graph too*, so the activation hump can be visually compared with and
without checkpointing.

---

## 2. What Phase 1 hands us

Phase 2 doesn't re-trace anything; it reads the already-built profiler:

| From `GraphProfiler` | What Phase 2 uses it for |
|---|---|
| `intermediates` — the list of `Intermediate(node, size_bytes, last_fwd_idx, first_bwd_idx, recompute_ms)` | the candidate set for eviction |
| `aliases_by_owner`, `node_size_bytes`, `node_type`, `idx`, `sep_*_idx` | the simulator (`simulate_peak`) |
| `avg_runtime_ms` | per-node recomputation cost |
| `nodes`, `region` | the `validate_recompute_set` and rewrite passes |

So the entire algorithm is a **function of the profiler's static state plus
the recompute timings**.  No new measurements needed.

---

## 3. The algorithm in one screen

```
peak_before = simulate_peak(profiler, evicted={})
mem_limit   = peak_before - total_activation_bytes / 2     # default target

evicted, order = ∅, []
loop:
    current_peak = simulate_peak(profiler, evicted)
    if current_peak <= mem_limit: break                    # target reached

    score every remaining candidate c:
      trial_peak = simulate_peak(profiler, evicted ∪ {c})
      if trial_peak >= current_peak: skip                  # doesn't help
      cost  = body_runtime_ms(recompute_body(c, retained))
      ratio = c.size_bytes / cost
      score = (ratio, peak_drop, c.size)

    if no candidate scored: break ("no remaining lowers peak")

    evict the highest-scoring c
    add c to order

to_recompute, to_retain = validate_recompute_set(profiler, order)
peak_after = simulate_peak(profiler, to_recompute)
extra_ms   = estimate_recompute_ms(profiler, to_recompute)
```

Two stop conditions: hit the memory target, or no remaining eviction
lowers the current peak (the peak isn't activation-dominated, so AC can't
help — see §6.3).

---

## 4. Component reference

`activation_checkpoint.py` (~430 lines).  Same documentation style as the
Phase 1 file: each piece explains **why** we need it, **the idea**, and
**how it flows** into the next thing.

### 4.1 `SelectionResult`

```python
@dataclass
class SelectionResult:
    to_recompute:           List[fx.Node]
    to_retain:              List[fx.Node]
    peak_before:            int
    peak_after:             int
    mem_limit:              Optional[int]
    estimated_recompute_ms: float
    reason:                 str           # why the loop stopped
```

**Why**: a single object that captures *what was decided* and *what trade-
off the decision implies*, so the harness can plot, the rewriter can act,
and the report can explain — all from one return value.

**Flow**: returned by `select_activations`, consumed by
`rewrite_with_checkpointing` (Phase 3) and `print_ac_decisions`.

### 4.2 `recompute_body(profiler, activation, retained)` — the boundary walk

```python
def recompute_body(profiler, activation, retained):
    placeholders = {n for n in profiler.nodes if n.op == "placeholder"}
    boundary     = placeholders | retained
    needed = set()
    stack  = [activation]
    while stack:
        n = stack.pop()
        if n in needed or n in boundary: continue
        needed.add(n)
        stack.extend(n.all_input_nodes)
    return [n for n in activation.graph.nodes if n in needed]
```

**Why**: the central question for every potential eviction is "what nodes
do I need to re-execute to rebuild this activation?"  The answer is the
forward subgraph that connects `activation` back to tensors that are
guaranteed to be live at backward time — placeholders (params, optimizer
state, batched inputs) plus retained activations.

**Idea**: BFS/DFS backwards through `all_input_nodes`, stopping at the
boundary set.  The result is a topologically-ordered list (we filter by
appearance in `activation.graph.nodes`, which is the FX graph's natural
topo order).

**Flow**: used by `body_runtime_ms` (cost), `estimate_recompute_ms`,
`validate_recompute_set`, `candidate_recompute_ms` inside the greedy
loop, AND by Phase 3's `rewrite_with_checkpointing` to know which nodes
to copy.  This is the single most-reused helper in the file.

### 4.3 `body_runtime_ms` and `estimate_recompute_ms`

```python
def body_runtime_ms(profiler, body):
    return sum(profiler.avg_runtime_ms.get(n.name, 0.0) for n in body)

def estimate_recompute_ms(profiler, to_recompute):
    selected = set(to_recompute)
    if not selected: return 0.0
    retained = {i.node for i in profiler.intermediates} - selected
    return sum(body_runtime_ms(profiler, recompute_body(profiler, n, retained))
               for n in sorted(selected, key=lambda n: profiler.idx[n]))
```

**Why**: AC's time cost is the runtime of the recomputation bodies it
adds to the backward pass.  We need a per-body cost (for the greedy's
ratio metric) and a total cost (for the latency comparison plot).

**Idea**: per-body cost = sum of measured per-node runtimes; total cost =
sum across all evicted activations, walking each one's body
*independently* (matching the additive cost model the rewrite uses).
The "independent" assumption is conservative — a smarter rewrite could
share work across overlapping bodies, but that's outside Phase 2.

**Flow**: `body_runtime_ms` feeds `candidate_recompute_ms` and
`estimate_recompute_ms`; the latter goes into `SelectionResult` and the
phase2 latency comparison.

### 4.4 `simulate_timeline_by_role` and `simulate_peak`

```python
def simulate_peak(profiler, evicted):
    timeline = simulate_timeline_by_role(profiler, evicted)
    return max(sum(timeline[role][t] for role in NodeType) for t in range(n))
```

**Why**: every greedy iteration needs to ask "if I evict candidate `c`,
what's the new peak?"  Without a fast simulator we'd be stuck doing
`O(candidates²)` graph rewrites and re-runs.

**Idea**: same shape as Phase 1's `memory_timeline_by_role` (walk by
storage owner, sum bytes per step) with **one extra rule**: if the owner
has any alias in `evicted` AND no other alias keeps the storage alive
across the forward/backward boundary, the owner is treated as live only
on the forward window plus a single backward step (the recomputation
spike at `first_bwd_idx`).

This captures both effects of an eviction:
- the storage *is* still allocated briefly during the forward pass (it
  has to be; that's where it's produced and used by other forward ops),
- and it briefly comes back at one backward step (where it's recomputed
  and immediately consumed).

**Flow**: called O(candidates²) times during the greedy.  Returns a peak
that the loop compares against `mem_limit`.

### 4.5 `validate_recompute_set` — drop unrecomputable selections

```python
def validate_recompute_set(profiler, selected_order):
    selected = set(selected_order)
    while changed:
        retained = all_acts - selected
        for node in selected:
            body = recompute_body(profiler, node, retained)
            valid = body and all(n.op in {"call_function", "call_method", "call_module"}
                                 and profiler.idx[n] < profiler.sep_idx
                                 for n in body)
            if not valid: selected.discard(node); changed = True
    ...
```

**Why**: the greedy can pick activations whose recomputation body
includes nodes that don't actually exist as forward call_functions
(usually because some ancestor wasn't reachable from a placeholder at all
— this happens with constants baked into the graph, or when an
intermediate's body would dip into the backward region).  Without this
pass we'd hand Phase 3 evictions it can't safely splice.

**Idea**: iterate to a fixed point.  Demoting one selection back to
"retained" can change another selection's body (it now has more retained
ancestors to rely on), so we re-check until nothing changes.

**Flow**: called once at the end of `select_activations`, after the
greedy loop.  Returns the cleaned `(to_recompute, to_retain)` lists.

### 4.6 `candidate_recompute_ms` — cost during the greedy

```python
def candidate_recompute_ms(profiler, candidate, evicted, all_activations, fallback_ms):
    retained = all_activations - evicted - {candidate}
    cost = body_runtime_ms(profiler, recompute_body(profiler, candidate, retained))
    return cost if cost > 0 else fallback_ms
```

**Why**: the µ-TWO ratio is `size / recompute_time`.  The recompute time
*depends on the current eviction set* — if we've already evicted `c`'s
ancestor `a`, then recomputing `c` requires recomputing `a` first, so
`c`'s cost grows.  This is the cascade.

**Idea**: build the body against `all_activations - evicted - {candidate}`
(everything still retained, minus the candidate itself) and sum runtimes.
The `fallback_ms` (the candidate's own `recompute_ms` from `Intermediate`)
is used if the body is empty for any reason.

**Flow**: called once per remaining candidate per greedy iteration.

### 4.7 `select_activations` — the µ-TWO greedy

The main entry point.  Already shown in §3.

**Why this scoring**: `(ratio, peak_drop, size)` ranks candidates in
order:
- **ratio** = bytes saved per millisecond of recomputation.  Highest
  ratio = most memory bang for compute buck.  This is the headline
  µ-TWO metric.
- **peak_drop** is a tie-breaker — given equal ratio, prefer the one
  that drops the peak more (some evictions help the peak more than
  others depending on lifetime overlap).
- **size** is the second tie-breaker, in case both above are equal.

**Why the cascade matters**: without the per-iteration cost
recomputation, the greedy would underestimate the price of evicting late-
in-the-chain activations and over-evict.  By rebuilding `cost` against
the *current* `evicted` set, ancestor costs are amortized correctly.

### 4.8 Phase 3: `rewrite_with_checkpointing`

```python
for act in sorted(to_recompute, key=lambda n: profiler.idx[n]):
    body = recompute_body(profiler, act, retain_set)
    with gm.graph.inserting_before(first_backward_user(act)):
        copies = {}
        for orig in body:
            new = gm.graph.node_copy(orig, arg_transform=lambda a: copies.get(a, a))
            copies[orig] = new
    new_act = copies[act]
    for user in list(act.users):
        if user is not new_act and profiler.idx.get(user, -1) >= profiler.sep_bwd_idx:
            user.replace_input_with(act, new_act)
gm.graph.lint(); gm.recompile()
```

**Why**: the selector's output is just a list of nodes; Phase 3 actually
edits the FX graph so the rewritten `gm` materializes those activations
on demand instead of holding them through the forward pass.

**Idea, per evicted activation `act`**:

1. **Compute the body** — same `recompute_body` as the simulator used.
   Because we walk against the same boundary, the rewrite's
   recomputation chain matches what the simulator priced.
2. **Splice copies into the backward** — insert a fresh copy of every
   body node right before `act`'s first backward user.  The
   per-iteration `copies: original → copy` map lets `node_copy`'s
   `arg_transform` rewire the copies to reference each other (and fall
   through to the original boundary nodes, which are still alive).
3. **Redirect later backward users** — every backward node that used
   `act` now uses the copy instead.  The original `act` keeps its
   forward users so the forward pass is unchanged.

After all evictions: lint and recompile.  We also erase any orphan
`detach` nodes the autograd tracer leaves behind — they're harmless but
trip `lint`.

**Important property**: each evicted activation gets its own
*independent* body.  No sharing across evictions.  This matches the
additive cost model the selector uses (see §4.6) — if the selector
costs `c1`'s body and then costs `c2`'s body, the rewrite must
materialize both bodies separately or the actual time wouldn't match
the estimate.

### 4.9 `first_backward_user`

A one-liner: the earliest backward-region user of an activation.  This
is the splice point for the recomputed copy — we want the copy to land
*just before* anyone in backward asks for it, and all other backward
users will find it via `replace_input_with`.

### 4.10 `print_ac_decisions`

A printed report: per-recompute and per-retain rows with size /
cost / ratio, then a summary block (target, peak before, peak after,
freed bytes, recomputed bytes, retained bytes, extra ms, stop reason).
This is what the Phase 2 entry point dumps to stdout for each
`(model, batch_size)` so the user can read what was picked at a glance.

---

## 5. End-to-end flow (`phase2.py`)

```
for (model, batch_size):
    profiler = GraphProfiler(gm); warm-up; measurement; aggregate_stats()
    selection = select_activations(profiler, mem_limit=...)

    # Plot the breakdown twice for the same profiler:
    plot_memory_breakdown(profiler, "no_ac.png")
    plot_memory_breakdown(profiler, "ac_est.png",
                          timeline_by_role=simulate_timeline_by_role(
                              profiler, selection.to_recompute))

    record (peak_no_ac, peak_ac_est, latency_no_ac, latency_ac_est)

for model:
    plot_peak_memory_vs_batch(rows, ...)         # grouped bars
    plot_latency_comparison_vs_batch(rows, ...)
```

The visualizer's `plot_memory_breakdown` accepts an optional
`timeline_by_role` argument — when supplied, it plots that timeline
instead of recomputing from the profiler.  This is how the same plotter
draws both the original chart and the AC-simulated chart from one
`GraphProfiler`.

`peak_ac_est` and `latency_ac_est` are *static estimates* — what the
selector and simulator say AC would achieve.  They aren't measured from
a rewritten graph; that's the Phase 3 validation step (`validate.py`
runs the rewrite and compares gradients).

---

## 6. Things that bit us

### 6.1 The cascade

First implementation took each candidate's `recompute_ms` directly from
the `Intermediate` dataclass — a constant, set once at profiling time.
The greedy then over-evicted late-chain activations because their
per-iteration true cost (which grows when ancestors are evicted) was
under-counted.  Fix: compute cost against the *current* eviction set
inside the loop (`candidate_recompute_ms`).

### 6.2 Selections without a body

A selection whose body would dip into the backward region (or contains
nodes our walker can't reach from a placeholder) isn't valid — the
rewrite would fail.  `validate_recompute_set` iterates a fixed-point
demotion until every remaining selection has a clean forward subgraph.
This catches a couple of pathological cases on the larger models.

### 6.3 When AC simply can't help

Two cases:
- **Peak is in the optimizer region.**  BERT bs=4/8 sits here — Adam's
  `m + v + denom` chain dominates (~2.1 GB) and the activation hump
  (~1.5 GB) is below it.  Evicting activations doesn't lower the peak.
  The greedy detects this when no candidate produces `peak_drop > 0`
  and bails with `reason="no remaining activation lowers the current
  peak"`.  The AC bar in the comparison chart then sits exactly on top
  of the no-AC bar — honest signal that AC isn't useful here.
- **Peak is dominated by parameters.**  Same story.  Phase 2 doesn't
  evict params (they're not in `intermediates`), so the bar can't drop
  below the param baseline regardless.

### 6.4 The simulator must match the rewriter

If the simulator and the rewriter compute different bodies, the static
"AC peak" estimate would diverge from what the rewritten graph
actually achieves.  Both call `recompute_body` with the same
`(placeholders | retained)` boundary — that's the shared contract that
keeps them aligned.

---

## 7. Files

| File | Purpose |
|---|---|
| [activation_checkpoint.py](activation_checkpoint.py) | the algorithm, the simulator, the rewriter, the printed report |
| [phase2.py](phase2.py) | Phase 2 entry point — sweep `(model, batch_size)`, save plots |
| [visualizer.py](visualizer.py) | the breakdown chart accepts an injected timeline; grouped bar charts for the comparison |
| [validate.py](validate.py) | Phase 3 validation: run the rewrite on a real `gm` and check gradients match the original |

To run:

```bash
python phase2.py                                  # sweep everything
python phase2.py resnet18 -b 8 16 32              # one model, three batch sizes
python phase2.py resnet50 -b 8 --mem-limit-mb 600 # explicit target
```

Outputs: per-`(model, bs)` `phase2_memory_<model>_bs<N>_no_ac.png` and
`_ac_est.png`; per-model `phase2_peak_vs_batch_<model>.png` and
`phase2_latency_vs_batch_<model>.png`.
