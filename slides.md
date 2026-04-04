---
title: "Activation Checkpointing"
subtitle: "CS265 Systems Project Midterm Check-in"
author: "Hadi Khalaf"
date: "Spring 2026"
theme: "Madrid"
colortheme: "dolphin"
fonttheme: "structurebold"
fontsize: 11pt
aspectratio: 169
header-includes:
  - \usepackage{booktabs}
  - \usepackage{tikz}
  - \usetikzlibrary{patterns}
  - \usepackage{ulem}
  - \setbeamertemplate{navigation symbols}{}
  - \definecolor{harvardcrimson}{RGB}{165,28,48}
  - \setbeamercolor{title}{fg=harvardcrimson}
  - \setbeamercolor{frametitle}{fg=harvardcrimson}
---

# Motivation

## Outline

\tableofcontents

## The Memory Problem in Deep Neural Net Training

A single training iteration requires **four main types** of GPU memory:

\begin{center}
\begin{tikzpicture}[scale=0.85, every node/.style={font=\small}]
  \draw[fill=blue!30] (0,0) rectangle (3,1) node[midway] {Parameters};
  \draw[fill=green!30] (0,1.1) rectangle (3,3.6) node[midway] {Activations};
  \draw[fill=orange!30] (3.2,0) rectangle (6.2,1.5) node[midway] {Gradients};
  \draw[fill=red!30] (3.2,1.6) rectangle (6.2,2.8) node[midway] {Optimizer State};
  \draw[fill=purple!20] (3.2,2.9) rectangle (6.2,3.6) node[midway] {Other};
  \node[anchor=west] at (6.5, 3.3) {\textbf{70--85\% of peak}};
  \draw[->, thick] (6.5, 3.1) -- (3.1, 2.5);
\end{tikzpicture}
\end{center}

Memory is the bottleneck limiting larger batch size and model scale

## Why Activations (Often) Dominate Peak Memory

\begin{center}
\begin{tikzpicture}[scale=0.7, every node/.style={font=\footnotesize}]
  \foreach \i/\name in {0/X, 1/Z1, 2/Z2, 3/Z3, 4/Z4, 5/L} {
    \draw[fill=green!20] (\i*2, 0) rectangle (\i*2+1.5, 0.6);
    \node at (\i*2+0.75, 0.3) {\name};
  }
  \foreach \i in {0,...,4} {
    \draw[->, thick] (\i*2+1.5, 0.3) -- (\i*2+2, 0.3);
  }
  \node[anchor=west] at (12, 0.3) {\textbf{Forward}};
  \foreach \i/\name in {5/dL, 4/dZ4, 3/dZ3, 2/dZ2, 1/dZ1} {
    \draw[fill=orange!20] (\i*2, -1.2) rectangle (\i*2+1.5, -0.6);
    \node at (\i*2+0.75, -0.9) {\name};
  }
  \foreach \i in {5,4,3,2} {
    \pgfmathtruncatemacro{\j}{\i-1}
    \draw[->, thick] (\i*2, -0.9) -- (\j*2+1.5, -0.9);
  }
  \node[anchor=west] at (12, -0.9) {\textbf{Backward}};
  \draw[->, dashed, red, thick] (1*2+0.75, 0) -- (1*2+0.75, -0.6);
  \draw[->, dashed, red, thick] (3*2+0.75, 0) -- (3*2+0.75, -0.6);
  \node[red, anchor=north] at (6, -1.5) {Activations must stay alive until backward uses them};
\end{tikzpicture}
\end{center}

- The forward pass produces Z1, ... while the backward  pass consumes them in reverse order. 
- **Example:** Z1 has the longest lifetime
- At the end of the forward pass, all activations are still in memory simultaneously.

## Activation Checkpointing: The Idea

**Idea: Trade compute for memory!** 

Discard selected activations after the forward pass and recompute during backward.

\begin{center}
\begin{tikzpicture}[scale=0.7, every node/.style={font=\footnotesize}]
  \node[anchor=east, font=\small\bfseries] at (-0.3, 0.3) {Without AC:};
  \foreach \i/\c in {0/green!30, 1/green!30, 2/green!30, 3/green!30} {
    \draw[fill=\c] (\i*2, 0) rectangle (\i*2+1.5, 0.6);
  }
  \node at (0.75,0.3) {Z1}; \node at (2.75,0.3) {Z2};
  \node at (4.75,0.3) {Z3}; \node at (6.75,0.3) {Z4};
  \node[anchor=west] at (8.5, 0.3) {All kept in memory};
  \node[anchor=east, font=\small\bfseries] at (-0.3, -1.1) {With AC:};
  \draw[fill=green!30] (0, -1.4) rectangle (1.5, -0.8);
  \draw[fill=red!15, dashed] (2, -1.4) rectangle (3.5, -0.8);
  \draw[fill=green!30] (4, -1.4) rectangle (5.5, -0.8);
  \draw[fill=red!15, dashed] (6, -1.4) rectangle (7.5, -0.8);
  \node at (0.75,-1.1) {Z1}; \node[red] at (2.75,-1.1) {Z2};
  \node at (4.75,-1.1) {Z3}; \node[red] at (6.75,-1.1) {Z4};
  \node[anchor=west] at (8.5, -1.1) {Z2, Z4 recomputed};
\end{tikzpicture}
\end{center}

- **Retain** expensive activations (convolutions)
- **Discard** cheap activations (ReLU, transpose)
- The question: **which to discard?** $\Rightarrow$ This is the idea behind Phase 2 ($\mu$-TWO algorithm)

## Project Overview

Three phases:

| Phase | Task | Status |
|-------|------|-------|
| 1 | **Graph Profiler**  | Done |
| 2 | **AC Selection** | Done |
| 3 | **Graph Rewriter** | Next |

\vspace{0.5em}

Models evaluated: **DummyModel**, **ResNet18**, **ResNet50**, **BERT-base**

# Phase 1: Graph Profiler

## Background: Why FX Tracing?

**Previous approach I had used:** PyTorch module hooks

- Must reconstruct graph, track tensor identity, and infer implicit autograd saves
- **Not rewritable**, i.e., can't insert recomputation nodes

\vspace{0.5em}

**Current approach:** `torch.fx` with `make_fx()`

- Traces the entire training step into a **single FX graph**
- Forward + backward + optimizer in one DAG
- Stable node names, explicit data dependencies via `node.users`
- Lifetimes directly readable from the graph
- **Rewritable** $\Rightarrow$ Phase 3 can modify the graph

## The Traced Graph Structure

\begin{center}
\begin{tikzpicture}[every node/.style={font=\small},
  block/.style={draw, rounded corners, minimum width=7cm, minimum height=0.6cm, fill=#1},
  arr/.style={->, thick, shorten >=2pt, shorten <=2pt}]
  \node[block=gray!15]  (ph)  at (0, 0)    {PLACEHOLDER --- params, opt states, batch};
  \node[block=green!15] (fwd) at (0, -1.0) {FORWARD --- t, addmm, relu, ...};
  \node[block=yellow!25](sep) at (0, -2.0) {\texttt{\%sep} --- end of forward};
  \node[block=purple!10](loss)at (0, -3.0) {LOSS --- sum, view, ones\_like};
  \node[block=yellow!25](sb)  at (0, -4.0) {\texttt{\%sep\_backward} --- start of backward};
  \node[block=orange!15](bwd) at (0, -5.0) {BACKWARD --- threshold\_backward, mm, ...};
  \node[block=red!10]   (opt) at (0, -6.0) {OPTIMIZER --- \_foreach\_add, copy\_, ...};
  \draw[arr] (ph) -- (fwd);
  \draw[arr] (fwd) -- (sep);
  \draw[arr] (sep) -- (loss);
  \draw[arr] (loss) -- (sb);
  \draw[arr] (sb) -- (bwd);
  \draw[arr] (bwd) -- (opt);
\end{tikzpicture}
\end{center}

`SEPFunction.apply(loss)` inserts the separator nodes during tracing.

## Boundary Detection

Scan for separator nodes to partition the graph:

```python
for i, node in enumerate(self.node_list):
    if node.target == torch.ops.separator.sep.default:
        self.sep_index = i        # end of forward
    elif node.target == torch.ops.separator.sep_backward.default:
        self.sep_bwd_index = i    # start of backward
```

Four regions: FORWARD, LOSS, BACKWARD, OPTIMIZER

## Parameter Identification

**Challenges:** (1) `fused=True` Adam crashes during FX tracing; (2) `foreach=True` has no single optimizer node; and (3) FakeTensors all have `requires_grad=False`.

\vspace{0.3em}

**Solution:** identify parameters by **usage pattern**:

```python
has_fwd_user = any(idx < sep_index ...)
has_opt_user = any(idx >= optimizer_index ...)
if has_fwd_user and has_opt_user:
    param_nodes.add(node)
```

| Placeholder type | Forward | Backward | Optimizer |
|-----------------|:-------:|:--------:|:---------:|
| **Parameter**   | Yes     | ---      | Yes       |
| Batch data      | Yes     | Yes      | ---       |
| Optimizer state | ---     | ---      | Yes       |

## Five Tensor Categories

Matching the course spec --- five categories:

| Category | How identified |
|----------|---------------|
| **PARAM** | Placeholder with forward + optimizer users |
| **ACT** | Forward `call_function` with backward users |
| **GRAD** | From `_fused_adam.args[1]` (fused mode only) |
| **OPTIMIZER\_STATE** | Placeholder with optimizer-only users |
| **OTHER** | Everything else |

## Intermediate Activations and Lifetimes

An **intermediate** satisfies four conditions:

1. `call_function` node (a computation)
2. Index $<$ `sep_index` (produced in forward)
3. Not a parameter
4. Has $\geq 1$ user with index $\geq$ `sep_bwd_index` (consumed in backward)

\vspace{0.5em}

For each intermediate:

- **`last_fwd_access`** = max index among forward users
- **`first_bwd_access`** = min index among backward users
- **Lifetime** = `first_bwd` $-$ `last_fwd` (idle period in steps)

Longer lifetime + larger size = better AC candidate.

## Runtime Profiling

For every node in `run_node()`:

```python
start_event.record()
result = super().run_node(n)
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
mem_delta = torch.cuda.memory_allocated() - mem_before
```

**Activation swapping** during profiling:

- After `last_fwd_access`: move to CPU (`.cpu()`)
- Before `first_bwd_access`: move back (`.cuda()`)
- Measures swap latency; prevents OOM on large models

## Memory Breakdown Visualization

\begin{center}
\begin{tikzpicture}[scale=0.55, every node/.style={font=\tiny}]
  \foreach \x/\p/\a in {0/0.5/0.5, 1/0.5/1, 2/0.5/1.5, 3/0.5/2, 4/0.5/2.5, 5/0.5/3, 6/0.5/3.5, 7/0.5/4, 8/0.5/3.5, 9/0.5/3, 10/0.5/2, 11/0.5/1, 12/0.5/0.5, 13/0.5/1, 14/0.5/2} {
    \draw[fill=blue!30] (\x*0.8, 0) rectangle (\x*0.8+0.7, \p);
    \draw[fill=green!30] (\x*0.8, \p) rectangle (\x*0.8+0.7, \p+\a);
  }
  \draw[dashed, red, thick] (7*0.8+0.35, 0) -- (7*0.8+0.35, 5);
  \node[red, anchor=south, font=\scriptsize] at (7*0.8+0.35, 5) {Peak};
  \draw[dotted, gray] (6*0.8, 0) -- (6*0.8, 4.8);
  \node[gray, anchor=south, font=\tiny] at (6*0.8, 4.8) {sep};
  \draw[dotted, gray] (8*0.8, 0) -- (8*0.8, 4.8);
  \node[gray, anchor=south, font=\tiny] at (8*0.8, 4.8) {sep\_bwd};
  \draw[->] (0, 0) -- (12.5, 0) node[right, font=\small] {Steps};
  \draw[->] (0, 0) -- (0, 5.3) node[above, font=\small] {MB};
  \draw[fill=blue!30] (1, -0.8) rectangle (1.5, -0.5); \node[anchor=west, font=\scriptsize] at (1.6, -0.65) {PARAM};
  \draw[fill=green!30] (4, -0.8) rectangle (4.5, -0.5); \node[anchor=west, font=\scriptsize] at (4.6, -0.65) {ACT};
  \draw[fill=orange!30] (7, -0.8) rectangle (7.5, -0.5); \node[anchor=west, font=\scriptsize] at (7.6, -0.65) {GRAD};
  \draw[fill=red!30] (10, -0.8) rectangle (10.5, -0.5); \node[anchor=west, font=\scriptsize] at (10.6, -0.65) {OPT\_STATE};
\end{tikzpicture}
\end{center}

Green accumulates in forward, freed in backward. Peak = where to apply AC.

## Phase 1 Results: Profiler Accuracy

| Model | Intermediates | Act. Mem | Est. Peak | GPU Peak | Ratio |
|-------|---:|---------:|----------:|---------:|------:|
| DummyModel | 19 | 4 MB | 6.5 MB | 14.5 MB | 0.45 |
| ResNet18 (bs=16) | 103 | 331 MB | 506 MB | 523 MB | **0.97** |
| ResNet50 (bs=16) | 268 | 1310 MB | 1643 MB | --- | --- |
| BERT-base (bs=4) | 363 | 716 MB | 2506 MB | 2118 MB | 1.18 |

\vspace{0.3em}

- DummyModel: CUDA allocator overhead dominates small tensors
- **ResNet18: 0.97x** --- near-perfect on large feature maps
- BERT: 1.18x overestimate from optimizer temporaries

# Phase 2: AC Selection Algorithm

## The Subset Selection Problem

**Input:** $N$ intermediates, each with size $m_i$ and recompute cost $c_i$

**Goal:** choose subset to discard $\Rightarrow$ minimise peak memory, bound extra compute

\vspace{0.5em}

This is **NP-hard** (reduces to knapsack).

\vspace{0.5em}

We use the greedy heuristic from **$\mu$-TWO** (Purandare et al., MLSys 2023).

## $\mu$-TWO: Background

A compiler for **concurrent multi-model training** on a single GPU.

Full algorithm considers both **swap** and **recompute**:

- **Swap candidate:** tensor with largest `inactive_time`
- **Recompute candidate:** tensor with largest `recompute_ratio`
- Each iteration: compare overhead, pick cheaper option

\vspace{0.5em}

**Our adaptation** for single-model AC:

- **Recompute branch only** (no CPU offloading)
- $\texttt{recompute\_ratio} = \texttt{memory\_size} / \texttt{total\_recomp\_time}$
- **Memory simulator** as stopping criterion

## The Greedy Algorithm (Algorithm B)

```
 1. baseline = simulate_peak(evicted={})
 2. mem_limit = baseline - 0.5 * total_act_mem
 3. if simulate_peak(ALL) >= baseline:
 4.     return  # AC cannot help
 5. while candidates remain:
 6.     if simulate_peak(evicted) <= mem_limit: break
 7.     if last eviction didn't reduce peak: break
 8.     pick candidate with max recompute_ratio
 9.     evict it
10.     propagate dependency changes
11. validate recompute set
```

Key: **memory simulator** decides when to stop, not a fixed budget.

## Memory Simulator

Estimates peak given evicted set:

\begin{center}
\begin{tikzpicture}[scale=0.65, every node/.style={font=\footnotesize}]
  % Normal tensor
  \draw[fill=green!30] (0, 1.2) rectangle (8, 1.8);
  \node at (4, 1.5) {Retained: live from produced to last\_use};
  % Evicted tensor
  \draw[fill=green!30] (0, 0) rectangle (2.5, 0.6);
  \draw[fill=white, pattern=north east lines, pattern color=red!30] (2.5, 0) rectangle (5.5, 0.6);
  \draw[fill=green!30] (5.5, 0) rectangle (6.2, 0.6);
  \node[font=\tiny] at (1.25, 0.3) {forward};
  \node[font=\tiny, red] at (4, 0.3) {FREE};
  \node[font=\tiny] at (5.85, 0.3) {recomp};
  \draw[<->] (2.5, -0.2) -- (5.5, -0.2) node[midway, below, font=\tiny] {idle period (saved)};
  \node[anchor=west] at (8.2, 0.3) {Evicted};
  \node[anchor=west] at (8.2, 1.5) {Retained};
\end{tikzpicture}
\end{center}

- Evicted tensor: free during idle period, briefly live at recomputation point
- Peak = max(sum of live tensors at each step)
- Called **after every eviction** to check if target met

## Cascading Recomputation (Alg. E/F)

If we evict A, and B needs A to be recomputed:

\begin{center}
\begin{tikzpicture}[scale=0.8, every node/.style={font=\small},
  box/.style={draw, rounded corners, minimum width=1.8cm, minimum height=0.6cm}]
  \node[box, fill=red!15] (a) at (0,0) {A (evicted)};
  \node[box, fill=green!15] (b) at (5,0) {B};
  \draw[->, thick] (a) -- (b) node[midway, above, font=\footnotesize] {recomp\_src};
  \node[anchor=north, font=\footnotesize, text width=5cm] at (5,-0.5)
    {B.recomp\_cnt += A.recomp\_cnt\\B.total\_time = B.time $\times$ B.cnt\\B.ratio $\downarrow$ (less attractive to evict)};
\end{tikzpicture}
\end{center}

Prevents cascading chains of expensive recomputations.

## Early Termination: The BERT Finding

**BERT-base** peak (2506 MB) at step 7366 --- deep in **optimizer region**.

No forward activations live there $\Rightarrow$ evicting them **cannot reduce peak**.

\vspace{0.3em}

```python
all_evicted_peak = simulate_peak(evicted=ALL)
if all_evicted_peak >= baseline_peak:
    return [], all_intermediates  # retain everything
```

\vspace{0.3em}

**Lesson:** BERT's bottleneck under `foreach=True` Adam is the optimizer, not activations. Would need optimizer sharding or `fused=True`.

## Validation: Input Availability

Every recomputed node's inputs must be **available**:

- Placeholder (always in memory), OR
- Retained intermediate

```python
while changed:
    for node in recompute_set:
        if any input also evicted:
            move to retain_set
            changed = True
```

Iterates until stable.

## Results: ResNet18

| | Before AC | After AC |
|---|---:|---:|
| Peak memory | 506 MB | **431 MB** |
| Recomputed | --- | 8 intermediates |
| Retained | 103 | 95 intermediates |
| Extra cost | --- | 1.05 ms/iter |

\vspace{0.3em}

- Recomputed: **ReLU outputs** (cheap, high ratio)
- Retained: **all 19 convolutions** (expensive, low ratio)
- Algorithm stops after 8 evictions --- simulator says target met

## Results: ResNet50 (bs=16)

| | Before AC | After AC |
|---|---:|---:|
| Peak memory | 1643 MB | **1129 MB** |
| Recomputed | --- | 16 intermediates |
| Retained | 268 | 252 intermediates |
| Extra cost | --- | 2.57 ms/iter |

\vspace{0.3em}

- **514 MB reduction** (31\% of peak) from just 16 evictions
- Recomputed: 3 early convolutions (50 MB, cheap), 6 relus, getitems, transpose
- Retained: all deeper convolutions (expensive, low ratio)
- Algorithm stops early --- simulator says target met after 16 ops

## Results: BERT-base

| | Before AC | After AC |
|---|---:|---:|
| Peak memory | 2506 MB | **2506 MB** |
| Recomputed | --- | **0** |
| Retained | 363 | **363** |
| Extra cost | --- | 0 ms |

\vspace{0.5em}

**AC cannot help.** Peak at optimizer step, not during forward/backward.

The algorithm correctly detects this and retains everything.

## Validation Results

Three-level validation (`validate.py`):

| Check | DummyModel | ResNet18 | BERT |
|-------|:----------:|:--------:|:----:|
| Profiler accuracy | 0.45x | **0.97x** | 1.18x |
| AC sanity | PASS | PASS | PASS |
| Simulator consistency | PASS | PASS | PASS |

55 unit tests, all pass on CUDA.

# Wrap-Up

## Key Findings

1. **AC is highly effective for vision models** --- cheap ops (ReLU, transpose) dominate recompute set; expensive ops (convolutions) correctly retained

2. **AC is ineffective for BERT with `foreach=True` Adam** --- peak is in optimizer region

3. **Profiler accuracy scales with tensor size** --- 0.45x for small tensors, 0.97x for large feature maps

4. **Memory simulator is essential** --- stops exactly when target met, avoids pointless eviction

5. **Cascading cost propagation matters** --- prevents compounding recomputation chains

## Next Steps: Phase 3

**Graph Rewriting** --- for each evicted activation:

1. Identify inputs (must be placeholders or retained)
2. Extract subgraph via `_extract_graph_with_inputs_outputs()`
3. Insert recomputation nodes before `first_bwd_access`
4. Replace backward uses with recomputed node

```python
with gm.graph.inserting_before(first_bwd_user):
    new_node = gm.graph.node_copy(n, arg_transform=...)
    replace_subsequent_uses_of(graph, old, new)
```

Correctness check: `torch.allclose(original_grads, checkpointed_grads)`

## Questions?

\begin{center}
\Large
Thank you!

\vspace{1em}
\normalsize
Code: \texttt{github.com/hskhalaf/cs265}

\vspace{0.5em}
\texttt{python starter\_code.py} \quad \texttt{python benchmarks.py Resnet18}

\vspace{0.5em}
\texttt{python validate.py --all}
\end{center}
