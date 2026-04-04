# Milestone 1

**Hadi Khalaf, CS2650, Spring 2026**

---

## 1. Introduction

During the forward pass, each layer produces an intermediate output called an activation that must be stored until the corresponding backward computation uses it for gradient calculation. Since the two passes run in opposite orders, the earliest activations remain in memory the longest, causing peak GPU memory to be dominated by this accumulated set rather than by the model weights. This project implements activation checkpointing (AC) for PyTorch. Instead of storing all activations, we keep only a chosen subset and recompute the others during the backward pass, trading extra computation for a lower memory peak. Milestone 1 builds the computation graph profiler that supports this system. 

We implemented it using PyTorch's module hook API, which instruments a live training iteration to build a DAG of every operation and tensor, classify each tensor by its role (parameter, activation, gradient, or optimizer state), and infer tensor lifetimes through a static analysis pass. Upon reviewing the course starter code, we found that PyTorch's FX tracing framework provides a better foundation: it captures the training step symbolically at compile time, producing a complete operator-level graph with stable tensor names and explicit forward/backward structure. This makes lifetimes directly readable from the graph and, crucially, makes the graph rewritable. This is what Milestones 2 and 3 need in order to insert recomputation nodes. Starting from Milestone 2, we will switch to the FX-based approach.

---

## 2. Problems Tackled

- **Hook-based graph construction:** observing each layer's inputs, outputs, and timing without modifying model code, while recovering a correct execution order across the forward and backward phases.
- **Tensor identity tracking:** associating the same logical tensor across multiple hook callbacks, when Python garbage collection recycles object addresses and the GPU allocator reuses buffer addresses.
- **Activation lifetime inference:** determining the true live interval of each activation, including tensors that PyTorch holds in internal C++ autograd closures invisible to Python hooks.
- **Optimizer state tracking:** detecting moment buffers that Adam allocates on its first step, which land at GPU addresses recently freed by gradient tensors.

---

## 3. Technical Description

### Problem 1 — Hook-based Graph Construction

**a) Problem framing.** PyTorch provides module hooks that fire before a layer runs (pre-hook), after it runs (post-hook), and after its backward computation. The problem is that container modules like `nn.Sequential` have their forward hook fire after all children's forward hooks but before any backward hooks, interleaving the two phases and making it impossible to recover a valid execution order.

**b) High-level solution.** We install hooks only on leaf modules since only they perform computation. A node ID is assigned at the forward pre-hook, so forward IDs are always smaller than backward IDs. Topological sort uses a min-heap variant of Kahn's algorithm keyed on node ID, which naturally schedules all forward nodes before all backward nodes without needing explicit edges between the two phases.

**c) Deeper details.** The pre-hook reserves a node ID and snapshots memory before the layer runs. The post-hook registers inputs and outputs in the tensor registry and adds a FORWARD node. The backward hook reserves a new (larger) node ID, registers gradient tensors, and adds a BACKWARD node. Timing and memory deltas are recorded at each step from the before/after snapshots.

---

### Problem 2 — Tensor Identity Tracking

**a) Problem framing.** A single logical tensor appears across multiple callbacks: `Linear`'s output shows up in its post-hook, in `ReLU`'s pre-hook, and in `Linear`'s backward hook. Python's `id(tensor)` is unreliable for tracking this: the Python wrapper can be garbage-collected while the GPU buffer stays alive, and the address gets recycled. Separately, the CUDA allocator reuses freed buffer addresses, so in networks with uniform layer widths a new activation can land at the same address and shape as a recently freed one.

**b) High-level solution.** We index the registry by `(data_ptr, shape)` instead of Python object identity. The GPU buffer address is stable while storage is alive, and pairing it with shape catches most allocator-reuse cases. For forward outputs, where reuse is most frequent, we always write a fresh entry. For other lookups, a role-mismatch check detects reuse and creates a fresh entry when the stored role conflicts with the expected one.

---

### Problem 3 — Activation Lifetime Inference

**a) Problem framing.** Accurate peak-memory estimation requires each tensor's live interval, i.e. when it is first produced and when it is last needed. Hook-visible data alone underestimates lifetimes because PyTorch saves certain forward tensors inside C++ autograd closures that Python never sees. For example, `Linear` keeps its forward input alive until the backward hook fires, since computing the weight gradient requires it. Ignoring these implicit saves leads to roughly a 2x underestimate of peak activation memory.

**b) High-level solution.** After graph construction we run a four-pass static analysis. Passes 1–2 set `first_use_op` and `last_use_op` from the hook-visible producers and consumers. Pass 3 pairs each forward node with its backward counterpart by module name and extends activation lifetimes to cover the implicit save: input activations for linear-like layers (the backward needs the forward input for `dL/dW`), output activations for element-wise layers like ReLU and Sigmoid (the backward needs the forward output for the chain rule). Pass 4 applies fixed rules for tensors whose lifetimes cannot be read from hooks: parameters span the full iteration, optimizer states persist to the end, and weight gradients extend to the optimizer step.

**c) Deeper details.** The key design decision in Pass 3 is the layer-type split. For linear-like layers (Linear, Conv, Embedding), the weight gradient computation `dL/dW = dL/dZ^T @ x` requires the forward input `x`, so we extend the input's lifetime. For activations like Sigmoid (`dL/dx = dL/dZ · Z · (1-Z)`), the derivative depends on the forward output `Z`, so we extend the output's lifetime instead. Without this distinction, Pass 3 would extend the wrong tensors and still underestimate peak memory.

---

### Problem 4 — Optimizer State Tracking

**a) Problem framing.** Adam maintains a first-moment and second-moment buffer per parameter, allocated on the first call to `optimizer.step()` with no hook API to observe it. By this point the backward pass has already freed the gradient tensors, and the CUDA allocator tends to place the new optimizer buffers at those same addresses. A naive registry lookup would find the old gradient entry and corrupt it.

**b) High-level solution.** We wrap the optimizer in a `WrappedOptimizer` that snapshots the set of `(param_id, state_key)` pairs in `optimizer.state` before calling `step()`, then compares after. Any newly appeared pair corresponds to a freshly allocated buffer, which we register as `OPTIMIZER_STATE` using `force_create`. We key the snapshot on `(param_id, state_key)` rather than buffer addresses precisely because those addresses may match recently freed gradients.

---

## 4. Challenges

- **AC subset selection (M2).** Choosing which activations to discard is NP-hard in general. The profiler provides sizes and recomputation costs, but designing a heuristic that makes good memory/computation tradeoffs is non-trivial, and its quality will directly determine the practical value of the system.
- **Subgraph extraction and rewriting (M3).** Switching to FX gives us a rewritable operator-level graph, making subgraph extraction tractable. The challenge is doing it correctly: identifying the exact subgraph for each discarded activation, cloning it into the backward pass at the right position, and ensuring the recomputed tensor replaces the original without disrupting other gradient computations.
- **Dynamic control flow.** The profiler captures a single iteration. Models with data-dependent graph structure, such as those using dynamic attention masks, may require a different checkpointing schedule on each step, which a static graph cannot support.
- **Scale.** The profiler has been tested on small feedforward networks. ResNet-152 and BERT introduce residual connections, parameter sharing, and many more nodes, which may surface correctness issues not seen at small scale.
