# CS265 Neural Network Profiler — Technical Report

## Table of Contents
1. [What Is This Profiler and Why Do We Need It?](#1-what-is-this-profiler-and-why-do-we-need-it)
2. [Background: How PyTorch Training Works](#2-background-how-pytorch-training-works)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Component Deep Dives](#4-component-deep-dives)
   - 4.1 [graph.py — Core Data Structures](#41-graphpy--core-data-structures)
   - 4.2 [memory.py — MemoryMonitor](#42-memorypy--memorymonitor)
   - 4.3 [tensor_registry.py — TensorRegistry](#43-tensor_registrypy--tensorregistry)
   - 4.4 [hooks.py — HookManager and WrappedOptimizer](#44-hookspy--hookmanager-and-wrappedoptimizer)
   - 4.5 [executor.py — ProfilerExecutor](#45-executorpy--profilerexecutor)
   - 4.6 [visualizer.py — MemoryVisualizer](#46-visualizerpy--memoryvisualizer)
5. [Design Challenges and How We Solved Them](#5-design-challenges-and-how-we-solved-them)
6. [Interpreting the Output](#6-interpreting-the-output)
7. [Known Limitations](#7-known-limitations)
8. [Test Suite](#8-test-suite)
   - 8.1 [Testing philosophy: what could go wrong?](#81-testing-philosophy-what-could-go-wrong)
   - 8.2 [Test categories](#82-test-categories)
   - 8.3 [Challenges discovered during testing](#83-challenges-discovered-during-testing)
   - 8.4 [Running the tests](#84-running-the-tests)

---

## 1. What Is This Profiler and Why Do We Need It?

Training a neural network involves much more memory than just storing the model's parameters (weights). During a single training iteration, the GPU must simultaneously hold:

- **Parameters** — the weight matrices W1, W2, …
- **Activations** — intermediate outputs at each layer (Z1, Z2, …), which must be kept alive throughout the backward pass because the gradient computation needs them
- **Gradients** — dL/dW for each parameter, accumulated by the backward pass
- **Optimizer state** — Adam keeps two extra buffers per parameter (the first and second moment estimates); SGD with momentum keeps one

Understanding *which* of these dominates memory usage *at which point in time* is critical for:
- Deciding how large a batch size you can afford
- Understanding why your model runs out of memory mid-backward
- Knowing where to apply memory-saving techniques like gradient checkpointing

This profiler instruments one full training iteration (forward → backward → optimizer step), builds a graph of every operation, categorizes every tensor by its role, and produces a timeline showing how much memory each category occupies at each step.

---

## 2. Background: How PyTorch Training Works

Before reading the code, you need to understand what PyTorch does during training.

### The Three Phases of Training

```
Forward Pass:   X → layer1 → Z1 → layer2 → Z2 → ... → output → loss
Backward Pass:  loss.backward() → grad flows backward through each layer
Optimizer Step: optimizer.step() → uses grads to update parameters
```

**Forward pass** — data flows left to right. Each layer takes an input, computes an output, and PyTorch records the computation in an *autograd graph* (a dynamic graph of which operations produced which tensors).

**Backward pass** — `loss.backward()` triggers reverse traversal of the autograd graph. The chain rule is applied at each node: each layer receives the gradient of the loss with respect to its output, and computes the gradient with respect to its input and its parameters. A key subtlety: **each layer must remember certain tensors from the forward pass**. For example:
- `Linear(x)` computes `Z = x @ W`. During backward, to compute `dL/dW = dL/dZ^T @ x`, the layer needs `x` (its forward input). So `x` is saved inside PyTorch's autograd closure and cannot be freed until after the backward hook fires for that layer.
- `Sigmoid(x)` computes `Z = 1 / (1 + exp(-x))`. The backward formula is `dL/dx = dL/dZ * Z * (1 - Z)`, which needs `Z` (the forward output). So Sigmoid saves its *output* for backward.

This is why activations have non-trivial lifetimes: they must stay alive far longer than just the next layer's forward pass.

**Optimizer step** — uses the gradients `param.grad` to update each parameter. Optimizers like Adam also maintain persistent state buffers (momentum estimates) that persist across training iterations.

### PyTorch Hooks

PyTorch allows you to attach *hooks* — callbacks that fire automatically at specific points during execution. Three hook types are relevant to us:

- `register_forward_pre_hook(fn)` — `fn` is called *before* a module's `forward()` runs. Receives the module and its inputs.
- `register_forward_hook(fn)` — `fn` is called *after* `forward()` completes. Receives the module, its inputs, and its outputs.
- `register_full_backward_hook(fn)` — `fn` is called after the module's contribution to the backward pass finishes. Receives grad_output (the incoming gradient) and grad_input (the outgoing gradient).

Our profiler installs all three hook types on every leaf module (a layer with no sub-layers) in the model. The hooks fire automatically during `model(X)` and `loss.backward()`, collecting timing and memory data as execution proceeds.

---

## 3. High-Level Architecture

The profiler is split into six files, each with a single clear responsibility:

```
profiler/
├── graph.py           # Data structures: what is a node? an edge? a tensor?
├── memory.py          # How to measure memory and time for one operation
├── tensor_registry.py # Track and classify every tensor seen during execution
├── hooks.py           # Install callbacks that capture data during training
├── executor.py        # Orchestrate the whole profiling run
└── visualizer.py      # Draw the memory breakdown chart
```

The execution flow is:

```
ProfilerExecutor.__init__()
  ├── Creates ComputationalGraph (empty)
  ├── Creates TensorRegistry (knows all model parameters from the start)
  ├── Creates MemoryMonitor
  ├── Creates HookManager → attaches hooks to all leaf modules
  └── Wraps the optimizer in WrappedOptimizer

ProfilerExecutor.run(X, Y, loss_fn)
  ├── model(X)            ← forward hooks fire, building graph nodes
  ├── loss_fn(output, Y)  ← loss computed (not hooked)
  ├── loss.backward()     ← backward hooks fire, adding more nodes
  ├── opt.step()          ← WrappedOptimizer fires, adding optimizer node
  ├── graph.freeze()      ← lock the graph against further changes
  └── compute_lifetimes() ← static analysis pass

ProfilerExecutor.get_report()  → ProfileReport (text summary)
ProfilerExecutor.visualize()   → PNG chart
```

Each component feeds the next. The registry tracks tensors; the hooks use the registry; the graph stores hooks' output; the visualizer reads the graph.

---

## 4. Component Deep Dives

### 4.1 `graph.py` — Core Data Structures

This file defines the vocabulary of the entire system. It contains no logic — only the data structures that everything else builds on.

#### Why we need a graph at all

A neural network's computation is naturally a directed acyclic graph (DAG). Each operation (a "node") takes some tensors as input and produces others as output. By representing computation this way, we can answer questions like: "which node produced the tensor that this other node consumes?" and "what is the topological order of operations?" — questions that are essential for lifetime analysis.

#### The key data structures

```python
class OpPhase(Enum):
    FORWARD    # Operations in the forward pass
    BACKWARD   # Operations in the backward pass
    OPTIMIZER  # The optimizer's parameter update step
```

```python
class TensorRole(Enum):
    PARAMETER       # Model weights (W1, W2, ...) — persistent, require_grad=True
    GRADIENT        # dL/dW or dL/dZ — produced by backward, consumed by optimizer
    ACTIVATION      # Intermediate forward outputs (Z1, Z2, ...) — temporary
    OPTIMIZER_STATE # Adam's m/v buffers — persistent across iterations
    OTHER           # Loss value, input X, target Y — don't fit above categories
```

```python
@dataclass
class TensorMeta:
    tensor_id: int        # Unique integer ID for this tensor
    name: str             # Human-readable: "linear1.output", "grad_fc1.weight"
    role: TensorRole
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    nbytes: int           # Total bytes = num_elements * bytes_per_element

    first_use_op: int     # Which step (in topo order) first produces this tensor
    last_use_op:  int     # Which step last needs this tensor (longest lived consumer)
```

The `first_use_op` and `last_use_op` fields are initially `None` and are filled in later by the `compute_lifetimes()` function. They define the *lifetime interval* of the tensor: it must be in memory at every step `t` satisfying `first_use_op <= t <= last_use_op`.

```python
@dataclass
class GraphNode:
    node_id: int
    op_name: str                  # e.g., "Linear.forward", "Sigmoid.backward"
    phase: OpPhase
    module_fqn: str               # Fully-qualified name, e.g., "encoder.layer1.linear"
    input_tensor_ids: List[int]   # Which tensors flow INTO this op
    output_tensor_ids: List[int]  # Which tensors flow OUT of this op
    profile: Optional[ProfileResult]  # Timing + memory measurements
```

```python
@dataclass
class GraphEdge:
    src_node_id: int   # The node that PRODUCES the tensor
    dst_node_id: int   # The node that CONSUMES the tensor
    tensor_id: int     # Which tensor flows along this edge
```

#### The topological sort — Kahn's algorithm with a min-heap

Once the graph is built, we need to process nodes in a valid execution order — every producer before all its consumers. The classic algorithm for this is Kahn's algorithm:

1. Compute in-degree (number of incoming edges) for every node.
2. Start with all nodes that have in-degree 0 (no dependencies).
3. Repeatedly: pick one zero-in-degree node, add it to the output, decrement the in-degree of all its children. When a child reaches 0, add it to the ready set.

The subtlety is **what to do when multiple nodes have in-degree 0 simultaneously**. This happens constantly: backward nodes have no explicit edges from the forward graph (PyTorch's autograd graph is separate from our hook-level graph). Without tie-breaking, Kahn's algorithm could interleave backward nodes between forward nodes.

Our solution: use a **min-heap** keyed by `node_id`. Since node IDs are assigned in the order hooks fire (forward hooks fire before backward hooks), forward nodes always have smaller IDs than backward nodes. The min-heap ensures we always pick the smallest available node_id first, which naturally puts all forward nodes before backward nodes before the optimizer step.

```python
heap = [nid for nid, deg in in_degree.items() if deg == 0]
heapq.heapify(heap)
while heap:
    nid = heapq.heappop(heap)   # smallest node_id among ready nodes
    order.append(self._nodes[nid])
    ...
```

---

### 4.2 `memory.py` — MemoryMonitor

This component answers: "how much memory did this one operation allocate?"

```python
class MemorySnapshot:
    gpu_allocated_bytes: int   # Bytes currently allocated on GPU
    gpu_reserved_bytes: int    # Bytes reserved (but possibly not allocated) by CUDA
    cpu_traced_bytes: int      # Python/C heap bytes tracked by tracemalloc
```

#### GPU measurement

`torch.cuda.memory_allocated()` returns the total GPU bytes currently in use by live tensors. By taking a snapshot before and after an operation, the delta tells us how much new memory that operation allocated (net).

```python
before = self.snapshot()        # before.gpu_allocated_bytes = A
<run operation>
after = self.snapshot()         # after.gpu_allocated_bytes = A + delta
delta = after.gpu_allocated - before.gpu_allocated
```

One complication: GPU operations are *asynchronous*. When you call `torch.mm(A, B)` on CPU, Python returns immediately while the GPU kernel is still running. If you snapshot memory before the kernel finishes, you get a wrong answer. The fix is `torch.cuda.synchronize()` — a blocking call that waits until all pending GPU work is done.

Additionally, `torch.cuda.reset_peak_memory_stats()` followed by `torch.cuda.max_memory_allocated()` captures the *peak* memory during an operation (which may differ from the net delta if the operation allocates a large temporary buffer and frees it before returning).

#### CPU measurement

CPU memory is tracked with Python's built-in `tracemalloc`. Important caveat: `tracemalloc` only tracks Python-level allocations and the C extensions that use Python's allocator. PyTorch's C++ backend uses its own caching allocator (CachingAllocator) which does *not* go through `tracemalloc`. So on CPU, the `cpu_traced_bytes` field underestimates actual memory usage. On GPU, the CUDA measurements are accurate.

---

### 4.3 `tensor_registry.py` — TensorRegistry

This is the most complex component. It answers: "what is this tensor, and have we seen it before?"

#### The core problem: tensor identity

Every tensor in PyTorch has a *Python object identity* (`id(tensor)`) and an *underlying storage* (`tensor.data_ptr()`). These are different:

- `id(tensor)` is the memory address of the *Python wrapper object*. Python reuses these addresses after objects are garbage-collected. A tensor can be garbage-collected even while its underlying C++ storage is still alive (e.g., if PyTorch holds an internal reference but drops the Python wrapper).

- `tensor.data_ptr()` is the address of the *raw data buffer* on the device. This address is stable as long as the tensor's storage is alive.

Because we see the same logical tensor multiple times (once in the forward pre-hook, once in the post-hook, once in the backward hook), we need a stable identifier. Using Python's `id()` failed in practice: the Python wrapper for Z1 (linear1's output) was garbage-collected before Z3 (linear2's output) was created, and Python reused the same memory address for the new wrapper. Our profiler would then look up Z3's id, find Z1's registry entry, and treat them as the same tensor — silently corrupting the graph.

Our solution: use `(data_ptr, shape)` as a composite lookup key:

```python
key = (tensor.data_ptr(), tuple(tensor.shape))
```

Why include shape? Because memory can be reused. After a tensor is freed, the allocator may give the same memory address to a completely different tensor. But a newly-allocated tensor for a different role (e.g., a gradient after an activation is freed) will likely have a different shape. Adding shape to the key catches most of these reuse cases.

#### Classification priority

When a tensor is first seen, we classify it:

```python
def _classify(self, tensor, hint) -> TensorRole:
    if tensor.data_ptr() in self._param_ptrs:
        return PARAMETER             # 1. Known parameter storage
    if tensor.data_ptr() in self._grad_ptrs:
        return GRADIENT              # 2. Known gradient storage
    if hint == GRADIENT:
        return GRADIENT              # 3. Caller says it's a gradient
    if hint is not None:
        return hint                  # 4. Any other explicit hint
    if tensor.grad_fn is not None:
        return ACTIVATION            # 5. Has an autograd history → activation
    return OTHER                     # 6. Fallback
```

We pre-populate `_param_ptrs` in `__init__` by iterating `model.parameters()` and recording every parameter's `data_ptr()`. This makes parameter detection fast and reliable (parameters are never re-classified).

#### Three creation methods

The registry exposes three creation methods, each for a different situation:

**`get_or_create(tensor, hint_role, hint_name)`** — standard lookup. Checks the `(data_ptr, shape)` dict. If found, returns the existing entry *unless* there's a role mismatch indicating memory reuse (e.g., a GRADIENT claiming the address of an old ACTIVATION — the old tensor was freed and memory reused, so we must create a fresh entry). If not found, classifies and registers.

**`force_create(tensor, role, name)`** — always creates a new entry, overwriting any stale `(data_ptr, shape)` mapping. Used for **all forward output tensors**. Why? In networks where many layers have the same output shape (e.g., a 5-layer network where every `Linear(32, 32)` produces shape `[batch, 32]`), PyTorch frequently reuses the freed memory of Z1 for Z3, Z5, etc. All these tensors have the same shape and the same `data_ptr()` in sequence. Without `force_create`, the second activation would find the first activation's entry and return it — the profiler would think Z3 is Z1.

**`mark_gradient(param, grad, param_name)`** — specifically handles weight gradients (`param.grad`). These are attached to parameter tensors by the backward pass. If we've already seen this gradient (perhaps with a short local name like `"grad_weight"`), we update its name to the fully-qualified name (like `"grad_fc2.weight"`), which is more informative. We always prefer longer names as they are more specific.

**`mark_optimizer_state(tensor, name)`** — specifically handles optimizer state buffers (Adam's `exp_avg`, `exp_avg_sq`; SGD's `momentum_buffer`). Always creates a fresh entry. Why? After `loss.backward()`, the gradient tensors (`param.grad`) are no longer needed by PyTorch's autograd machinery. The optimizer allocates new buffers. SGD's `torch.clone(grad)` for the momentum buffer allocates at the same address as the (now-freed) gradient. Without fresh creation, `mark_optimizer_state` would find the old gradient entry and mutate it from `GRADIENT` to `OPTIMIZER_STATE` — silently destroying the gradient record.

#### The lifetime computation: `compute_lifetimes(graph, registry)`

After the graph is fully built and frozen, we need to determine for each tensor: what is the first operation that produces it, and what is the last operation that needs it?

The naive approach — only tracking explicit hook-visible usage — is wrong for activations. As discussed in the background section, PyTorch's autograd saves certain tensors inside C++ closures that are invisible to our Python hooks. We must infer these implicit uses from the network structure.

We use a **four-pass** algorithm (plus a fixup step):

**Pass 1 — First producer.** Walk nodes in topological order. For every tensor in a node's `output_tensor_ids`, if `first_use_op` is not yet set, set it to the current step index.

```python
for seq, node in enumerate(nodes):
    for tid in node.output_tensor_ids:
        meta = registry._registry.get(tid)
        if meta is not None and meta.first_use_op is None:
            meta.first_use_op = seq
```

**Pass 2 — Last explicit consumer.** Walk again. For every tensor in a node's `input_tensor_ids`, update `last_use_op` to be the maximum of its current value and the current step.

```python
for seq, node in enumerate(nodes):
    for tid in node.input_tensor_ids:
        meta = registry._registry.get(tid)
        if meta is not None:
            if meta.last_use_op is None or seq > meta.last_use_op:
                meta.last_use_op = seq
```

**Pass 3 — Forward-backward pairing (the crucial step).** This pass captures the implicit autograd saves. For each forward node, find its corresponding backward node (matched by `module_fqn`, the fully-qualified module name). Then:

- If the forward op is *linear-like* (Linear, Conv, Bilinear, Embedding): its backward needs the **input** activation (to compute the weight gradient). Extend all input tensor lifetimes to the backward step.
- Otherwise (activation functions like ReLU, Sigmoid, Tanh): their backward needs the **output** of the forward pass (as a mask or in the derivative formula). Extend all output tensor lifetimes to the backward step.

```python
fqn_to_bwd_seq = {}
for seq, node in enumerate(nodes):
    if node.phase == BACKWARD:
        fqn_to_bwd_seq[node.module_fqn] = seq   # map "linear1" -> step 7

for seq, node in enumerate(nodes):
    if node.phase != FORWARD:
        continue
    bwd_seq = fqn_to_bwd_seq.get(node.module_fqn)
    if bwd_seq is None:
        continue
    is_linear_like = any(s in node.op_name.lower()
                         for s in ("linear", "conv", "bilinear", "embedding"))
    tids_to_extend = (node.input_tensor_ids if is_linear_like
                      else node.output_tensor_ids)
    for tid in tids_to_extend:
        meta = registry._registry.get(tid)
        if meta is not None and meta.role == ACTIVATION:
            if meta.last_use_op is None or bwd_seq > meta.last_use_op:
                meta.last_use_op = bwd_seq
```

Without Pass 3, the lifetime analysis would underestimate how long activations live and would misidentify the memory peak.

**Pass 4 — Normalize persistent tensor lifetimes.** Passes 1–3 treat all tensors symmetrically, which produces incorrect lifetimes for tensors whose memory semantics don't match the hook-visible usage pattern:

- **PARAMETER**: parameters appear in the optimizer step's `output_tensor_ids` (where the update writes back), but they're live for the *entire* iteration — they must be in memory for every forward and backward computation. Pass 1 sets `first_use_op` to whenever they're first seen in a forward node, which may not be step 0. We normalize: `first_use_op = 0`, `last_use_op = n_steps - 1`.

- **OPTIMIZER_STATE**: Adam's `exp_avg`/`exp_avg_sq` are allocated during `step()` and persist across iterations. They should appear on the chart from the optimizer step to the end of the timeline. We set `last_use_op = n_steps - 1`.

- **Terminal gradients** (`param.grad`): weight gradients are produced during backward but consumed by the optimizer — yet no hook explicitly records this consumption as an input. After the Fixup step, terminal gradients have `last_use_op == first_use_op`. We extend them to `opt_seq` (the optimizer step). Intermediate gradients (the `dL/dZ` flow tensors consumed as input by the next backward node) already have `last_use_op > first_use_op` and are left alone.

```python
opt_seq = next((seq for seq, n in enumerate(nodes) if n.phase == OPTIMIZER), None)
for meta in registry._registry.values():
    if meta.first_use_op is None:
        continue
    if meta.role == PARAMETER:
        meta.first_use_op, meta.last_use_op = 0, n_steps - 1
    elif meta.role == OPTIMIZER_STATE:
        meta.last_use_op = n_steps - 1
    elif meta.role == GRADIENT and opt_seq is not None:
        if meta.last_use_op == meta.first_use_op and opt_seq > meta.last_use_op:
            meta.last_use_op = opt_seq
```

Without Pass 4, the static chart shows parameters as a thin sliver at the optimizer step instead of a constant blue baseline, optimizer states appear then immediately vanish, and terminal gradients disappear before reaching the optimizer step — all three are visually wrong.

---

### 4.4 `hooks.py` — HookManager and WrappedOptimizer

This is where data collection happens. Hooks are callbacks registered with PyTorch that fire automatically at defined points during execution.

#### Why only leaf modules?

```python
def attach(self, model: nn.Module) -> None:
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue   # Skip container modules
        ...
```

`model.named_modules()` returns *all* modules in the hierarchy: the top-level model, any sub-containers, and the leaf layers (Linear, ReLU, etc.). A container module's forward hook fires *after all of its children's hooks*, but *before the backward hooks for those same children*. This interleaves the execution order in a way that breaks topological sorting. Since container modules do no computation themselves (they just call their children in sequence), hooking them adds no useful information and corrupts the ordering. We skip them.

#### The forward pre-hook and `_pending`

The pre-hook fires before the forward pass of each layer:

```python
def hook(module, inputs):
    node_id = self._graph.next_node_id()   # reserve an ID for this node
    before = self._monitor.snapshot()
    t0 = time.perf_counter()
    self._pending[id(module)] = (node_id, t0, before)
```

We store `(node_id, t0, before)` in a dict keyed by `id(module)`. The post-hook will retrieve this to compute elapsed time and memory delta. Using `id(module)` as the key is safe here (unlike for tensors) because module Python objects are persistent — they live for the entire model lifetime and are never garbage-collected during a forward pass.

Why reserve the node ID in the pre-hook rather than the post-hook? Because node IDs are assigned sequentially, and we want the node_id to reflect the order in which layers *start* executing (DFS pre-order = the forward execution order), not when they finish.

#### The forward post-hook

The post-hook fires after `forward()` returns, with access to the layer's inputs and outputs:

```python
def hook(module, inputs, outputs):
    node_id, t0, before = self._pending.pop(id(module))
    t1 = time.perf_counter()
    after = self._monitor.snapshot()

    # Register inputs (look them up by data_ptr+shape)
    in_ids = [registry.get_or_create(t).tensor_id for t in _flatten_tensors(inputs)]

    # Register module parameters explicitly as PARAMETER
    for pname, param in module.named_parameters(recurse=False):
        registry.get_or_create(param, TensorRole.PARAMETER, pname)

    # Register outputs as fresh ACTIVATION entries
    out_ids = []
    for i, t in enumerate(_flatten_tensors(outputs)):
        role = ACTIVATION if t.grad_fn is not None else OTHER
        name = f"{module_name}.out{i}" if i > 0 else f"{module_name}.output"
        meta = registry.force_create(t, role, name)   # always fresh
        out_ids.append(meta.tensor_id)

    # Build the GraphNode
    node = GraphNode(node_id, op_name, FORWARD, fqn, in_ids, out_ids, profile)
    graph.add_node(node)

    # Add edges from producer nodes to this node
    for in_id in in_ids:
        src = self._find_producer(in_id)
        if src != -1:
            graph.add_edge(GraphEdge(src, node_id, in_id))
```

The key decision here is `force_create` for outputs. As explained above, activations in uniform-width networks reuse the same memory addresses. By always creating a fresh registry entry for each forward output, we guarantee that each layer's output gets its own unique `tensor_id`, regardless of memory reuse.

Why check `t.grad_fn is not None` to decide if an output is an ACTIVATION? In PyTorch, any tensor produced by an operation on tensors that require gradients carries a `grad_fn` — a pointer to the autograd function that produced it. A tensor without `grad_fn` was either a leaf (input/parameter) or was produced with `torch.no_grad()`. So `grad_fn is not None` is a reliable indicator of "this tensor was computed and can backpropagate through."

#### The backward hook

```python
def hook(module, grad_input, grad_output):
    # grad_output: gradients coming INTO this module (from the layer above in forward)
    # grad_input:  gradients going OUT  of this module (to the layer below in forward)

    in_ids = [registry.get_or_create(g, GRADIENT).tensor_id
              for g in _flatten_tensors(grad_output)]
    out_ids = [registry.get_or_create(g, GRADIENT).tensor_id
               for g in _flatten_tensors(grad_input) if g is not None]

    # Also capture weight gradients (param.grad is set by the backward pass)
    for pname, param in module.named_parameters(recurse=False):
        if param.grad is not None:
            meta = registry.mark_gradient(param, param.grad, pname)
            if meta.tensor_id not in out_ids:
                out_ids.append(meta.tensor_id)

    node = GraphNode(..., BACKWARD, ...)
    graph.add_node(node)
```

The naming of `grad_input` and `grad_output` is confusing (it refers to gradients w.r.t. the module's inputs/outputs in the *forward* direction, not the direction of gradient flow). Concretely:
- `grad_output` = the gradient signal arriving at this module = `dL/d(module_output)` = the "upstream" gradient
- `grad_input` = the gradient this module passes backward = `dL/d(module_input)` = the "downstream" gradient

#### WrappedOptimizer

The optimizer doesn't use hooks — it's not a `nn.Module`. Instead, we wrap it:

```python
class WrappedOptimizer:
    def step(self, closure=None):
        before = monitor.snapshot()
        t0 = time.perf_counter()

        pre_state = self._collect_state_ptrs()   # snapshot optimizer state keys
        result = self._opt.step(closure)          # run the real optimizer
        post_state = self._collect_state_ptrs()   # snapshot again

        t1 = time.perf_counter()
        after = monitor.snapshot()

        # New (param_id, key) pairs in post_state but not in pre_state
        # = newly allocated state tensors (Adam's lazy init)
        for param_id, key, tensor in post_state:
            if (param_id, key) not in pre_keys:
                registry.mark_optimizer_state(tensor, f"opt_{key}")
        ...
```

Adam uses *lazy initialization*: the first time `step()` is called, it allocates `exp_avg` and `exp_avg_sq` for each parameter. By diffing the optimizer's `state` dict before and after `step()`, we detect exactly which state tensors were newly created.

---

### 4.5 `executor.py` — ProfilerExecutor

This is the user-facing entry point. It orchestrates everything:

```python
profiler = ProfilerExecutor(model, optimizer, device)
profiler.run(X, Y, nn.MSELoss())
report = profiler.get_report()
report.print_summary()
profiler.visualize("memory_breakdown.png")
```

The `run()` method is intentionally a one-shot operation:

```python
def run(self, X, Y, loss_fn):
    if self._ran:
        raise RuntimeError("Already ran. Create a new instance for a new run.")

    # Register X and Y as OTHER (not parameters, not activations)
    registry.get_or_create(X, OTHER, "X")
    registry.get_or_create(Y, OTHER, "Y")

    # Forward (hooks fire automatically)
    output = model(X)

    # Loss (not hooked — we just register the loss tensor)
    loss = loss_fn(output, Y)
    registry.get_or_create(loss, OTHER, "L")

    # Backward (backward hooks fire automatically)
    loss.backward()

    # Register weight gradients (hooks may have missed them for bottom-layer modules)
    for name, param in model.named_parameters():
        if param.grad is not None:
            registry.mark_gradient(param, param.grad, name)

    # Optimizer step (WrappedOptimizer fires)
    opt.step()

    # Freeze and analyze
    graph.freeze()
    compute_lifetimes(graph, registry)
    hooks.detach()
    self._ran = True
```

Why register weight gradients again after `loss.backward()`? The `register_full_backward_hook` for a module fires when that module's backward computation is *complete*, but "complete" means the module has computed `dL/d_input` (which it passes upstream) — not necessarily that `param.grad` has been written. For leaf modules at the very bottom of the network (where the upstream gradient is wrt the raw inputs), PyTorch may not write `param.grad` until after the hook fires. The post-backward explicit loop catches these stragglers.

`graph.freeze()` locks the graph against further modifications. `compute_lifetimes()` does the four-pass lifetime analysis. `hooks.detach()` removes all hooks from the model (important to avoid memory leaks or double-firing in subsequent forward passes).

---

### 4.6 `visualizer.py` — MemoryVisualizer

Two modes of visualization, both producing a stacked bar chart.

#### Static mode (default)

For each time step t (0, 1, …, N-1), we iterate every tensor in the registry and add its `nbytes` to the appropriate role's bar if the tensor is live at that step (`first_use_op <= t <= last_use_op`):

```python
for meta in registry._registry.values():
    if meta.first_use_op is None:
        continue
    first = meta.first_use_op
    last = meta.last_use_op
    data[meta.role][first : last + 1] += meta.nbytes
```

This is a *formula-based estimate*: it uses the tensor's declared size (`nbytes = numel * element_size`) rather than measured GPU allocation. It's accurate as long as PyTorch doesn't apply hidden layout transformations (which it generally doesn't for standard Linear/Conv operations).

The result is a stacked bar at each step showing how much memory each role contributes. The peak step is marked with a dashed vertical line.

#### Dynamic mode

Uses the measured `gpu_memory_bytes` delta from each node's `ProfileResult`. Takes the cumulative sum over time to show the running total. This is only meaningful on GPU (CPU deltas are 0 due to the tracemalloc limitation). On CPU, static mode is more informative.

#### Chart layout

The x-axis is every operation in topological order. The y-axis is memory in MB. Bars are color-coded:
- Blue = PARAMETER
- Green = ACTIVATION
- Orange = GRADIENT
- Red = OPTIMIZER_STATE
- Purple = OTHER

---

## 5. Design Challenges and How We Solved Them

### Challenge 1: Container module hooks fire out of order

**Symptom:** The graph showed the TwoLayerMLP's own forward node inserted between the backward nodes of its children, because the container's post-hook fires last among the forward hooks but before any backward hooks.

**Root cause:** `model.named_modules()` returns the model itself alongside its children. The model's forward hook fires after all leaf modules have finished their forward passes, but its node_id is assigned in this late position, making it sort after backward nodes.

**Fix:** Skip any module that has children:
```python
if len(list(module.children())) > 0:
    continue
```
Container modules do no computation of their own — they merely call their children. Hooking only leaf modules loses no information.

---

### Challenge 2: Backward nodes interleaved with forward nodes in topological sort

**Symptom:** Topo order showed backward nodes appearing between forward nodes.

**Root cause:** Backward nodes have no incoming edges in our graph (we don't explicitly model the dependency "backward of layer N must run after forward of layer N"). Kahn's algorithm treats zero-in-degree nodes as equally "ready," so it might pick a backward node before all forward nodes are processed.

**Fix:** Replace the BFS queue with a min-heap keyed by `node_id`. Node IDs are assigned in hook-fire order, and forward hooks always fire before backward hooks. So forward nodes always have smaller IDs than backward nodes. The min-heap picks the smallest available ID, ensuring the forward-then-backward-then-optimizer order even without explicit edges.

---

### Challenge 3: Python `id()` reuse corrupting tensor lookup

**Symptom:** The profiler was treating Z3 (linear2's output) as the same tensor as Z1 (linear1's output), because they happened to share the same Python object ID.

**Root cause:** Python's memory allocator reuses object addresses. When Z1's Python wrapper was garbage-collected (PyTorch dropped its internal reference after passing it to Sigmoid), the memory address was freed back to Python's allocator. When Z3 was created, it was allocated at that same address, giving it the same `id()` as the late Z1.

**Fix:** Use `(data_ptr(), shape)` as the composite lookup key. The raw data pointer (`data_ptr()`) refers to the C-level storage, which remains stable as long as the tensor is logically alive (even after the Python wrapper is gone). This is much more stable than Python's `id()`.

---

### Challenge 4: Memory reuse causing misclassification

**Symptom:** A gradient tensor would be classified as ACTIVATION, or an optimizer state as GRADIENT, because the new tensor's `(data_ptr, shape)` matched an old, freed tensor's entry.

**Root cause:** After the forward pass, activations are freed one by one as the backward pass consumes and no longer needs them. The GPU allocator reuses their memory for gradient tensors. If the new gradient has the same shape as the freed activation (very common for per-layer gradients), `(data_ptr, shape)` lookup would find the old ACTIVATION entry.

**Fix (two-pronged):**

1. In `get_or_create()`, detect role mismatches: if the hint says GRADIENT but the existing entry is ACTIVATION/OTHER, treat this as memory reuse and create a fresh entry.

2. `mark_optimizer_state()` and `force_create()` always create fresh entries, overwriting any stale `(data_ptr, shape)` mapping. This is the correct behavior when we *know* a new tensor is being born (forward output or optimizer state allocation).

---

### Challenge 5: SGD momentum buffer overwriting gradient entries

**Symptom:** After the optimizer step, some GRADIENT entries had their role changed to OPTIMIZER_STATE, corrupting the categorization summary.

**Root cause:** `mark_optimizer_state()` originally called `get_or_create()`. SGD's momentum buffer is allocated via `torch.clone(param.grad)`, which often gets the same memory address as the (just-freed) gradient it was cloning. `get_or_create()` would find the gradient's entry and return it, then we'd mutate its `role` field.

**Fix:** `mark_optimizer_state()` never calls `get_or_create()`. It always creates a new `TensorMeta` directly and writes it into both `_registry` and `_ptr_shape_to_meta`, overwriting the stale gradient entry:

```python
def mark_optimizer_state(self, tensor, name):
    tid = self._next_id
    self._next_id += 1
    meta = TensorMeta(tid, name, OPTIMIZER_STATE, ...)
    self._registry[tid] = meta
    self._ptr_shape_to_meta[key] = meta   # overwrite
    return meta
```

---

### Challenge 6: Gradient names not fully qualified

**Symptom:** Weight gradients appeared as `grad_weight` instead of `grad_fc2.weight`, making it hard to tell which layer they belonged to.

**Root cause:** The backward hook fires per-module and sees only the local parameter name (`weight` or `bias`). The hook registered the gradient with the local name. The executor's post-backward loop tried to re-register with the fully-qualified name (`fc2.weight`), but the original condition `if not meta.name.startswith("grad")` prevented the update if the entry already had a gradient name.

**Fix:** In `mark_gradient()`, always prefer the *longer* name (longer = more qualified):
```python
if len(grad_name) >= len(meta.name):
    meta.name = grad_name
```
This upgrades `"grad_weight"` (10 chars) to `"grad_fc2.weight"` (15 chars) when the more qualified name is available.

---

### Challenge 7: Activation lifetime underestimation

**Symptom:** The memory chart showed the peak occurring at the optimizer step, not during the backward pass. For the 2-layer MLP, Z2 (sig1's output) appeared to have `last_use_op = 2` (the step when it flows into linear2.forward), but it actually needed to stay alive until step 6 (sig1.backward).

**Root cause:** PyTorch's autograd saves certain tensors *inside C++ closures* that are entirely invisible to our Python hooks. When `Sigmoid.forward(Z1)` produces Z2, PyTorch internally saves Z2 so that `Sigmoid.backward` can compute `dL/dZ1 = dL/dZ2 * Z2 * (1 - Z2)`. This saved reference keeps Z2 alive, but no Python hook sees this "use." Our Pass 2 analysis only saw Z2 consumed explicitly at step 2 (linear2.forward).

**Fix:** Pass 3 in `compute_lifetimes()`. By pairing each forward node with its corresponding backward node (matched by `module_fqn`), we can *infer* the implicit saves:
- Non-linear layers (Sigmoid, ReLU, Tanh, Dropout, …) save their **output** for backward → extend output lifetime to backward step.
- Linear-like layers save their **input** for the weight gradient → extend input lifetime to backward step.

This makes the lifetime estimates match the true memory usage pattern.

---

### Challenge 8: Same-shape activations sharing lookup entries

**Symptom:** In a 5-layer network where every layer has the same hidden size (e.g., `Linear(32, 32)`), only 3 of the expected 10 activations appeared in the registry. The other 7 were silently merged with earlier activations.

**Root cause:** After Z1 (layers.0.output, shape [batch, 32]) was freed during the backward pass, layers.1.output (Z3) was allocated at the same memory address with the same shape. `get_or_create()` found Z1's entry at key `(data_ptr, (batch, 32))` and returned it — treating Z3 as Z1.

**Fix:** All forward output tensors use `force_create()` instead of `get_or_create()`. `force_create()` unconditionally allocates a new `TensorMeta` and overwrites the stale `(data_ptr, shape)` mapping:

```python
# In _make_fwd_post:
meta = registry.force_create(t, role, name)   # <-- always fresh
```

This guarantees that each layer's output gets a unique registry entry, regardless of how PyTorch manages the underlying memory.

---

## 6. Interpreting the Output

### The Operation Summary Table

```
#    Op                             Phase      Module                      Time(ms)       Mem(B)
0    Linear.forward                 FORWARD    linear1                        0.073           +0
1    Sigmoid.forward                FORWARD    sig1                           0.038           +0
...
4    Sigmoid.backward               BACKWARD   sig2                           0.010           +0
...
8    Adam.step                      OPTIMIZER  optimizer                      0.250           +0
```

- **#** — Topological order index. All FORWARD nodes come first, then BACKWARD, then OPTIMIZER.
- **Op** — `ClassName.forward/backward` of the PyTorch module.
- **Phase** — FORWARD, BACKWARD, or OPTIMIZER.
- **Module** — the fully-qualified name as defined in your `nn.Module`. For nested modules, this would be something like `encoder.layer1.linear`.
- **Time(ms)** — wall-clock time for this one operation. On CPU, this includes Python overhead. On GPU, this is synchronized wall time.
- **Mem(B)** — net change in GPU memory allocated during this operation. `+0` on CPU means no change was measured (expected, due to tracemalloc limitations).

### The Tensor Categorization Summary

```
PARAMETER (2):
  weight   shape=torch.Size([8, 4])   0.12 KB
  weight   shape=torch.Size([2, 8])   0.06 KB

ACTIVATION (4):
  linear1.output   shape=torch.Size([16, 8])   0.50 KB
  sig1.output      shape=torch.Size([16, 8])   0.50 KB
  ...
```

- **PARAMETER** — one entry per weight matrix or bias vector. Count should equal the number of `nn.Parameter` objects in your model.
- **ACTIVATION** — one entry per `module.output` in the forward pass. Count equals the number of leaf modules (since each leaf produces exactly one output tensor, usually).
- **GRADIENT** — includes both weight gradients (`grad_fc1.weight`) and intermediate flow gradients (`grad_0`, `grad_1`, …). Flow gradients are the `dL/dZ` tensors that propagate backward through the network.
- **OPTIMIZER_STATE** — for Adam: `step` (scalar counter), `exp_avg` (first moment), `exp_avg_sq` (second moment) per parameter. 3 entries per parameter = 6 entries for a 2-parameter model.
- **OTHER** — input data X, target Y, loss scalar L. Not involved in the model's parameter update logic.

### The Activation Lifetime Table

```
Name                       First Use   Last Use   Live Ops   Size(KB)
linear1.output                     0          1          2       0.50
sig1.output                        1          6          6       0.50
linear2.output                     2          3          2       0.12
sig2.output                        3          4          2       0.12
```

- **First Use** — the step index when this activation is first produced.
- **Last Use** — the step index of the last operation that needs this activation (including implicit autograd saves from Pass 3).
- **Live Ops** — `Last Use - First Use + 1`. The number of steps during which this tensor must be held in memory.
- **Size(KB)** — memory footprint. `numel * element_size` (usually 4 bytes per float32).

**Key insight:** `sig1.output` (Z2) has `Live Ops = 6`, spanning steps 1 through 6. It stays alive from when Sigmoid produces it (step 1) until Sigmoid's backward hook fires (step 6), because `Sigmoid.backward` needs Z2 to compute the derivative (`Z2 * (1 - Z2)`). The larger the `Live Ops` span and `Size`, the more this activation contributes to the peak memory.

**Identifying the memory bottleneck:** Look for activations with large `Size` AND large `Live Ops`. These are the candidates for gradient checkpointing — you can recompute them during the backward pass instead of keeping them in memory, trading compute for memory.

### The Memory Breakdown Chart

The PNG chart shows a stacked bar at each operation step. The height of the total bar shows total memory in use at that point. Colors correspond to tensor roles.

**What to look for:**
- The peak (marked with a dashed line) tells you at which step the model uses the most memory.
- The composition of the peak bar tells you what dominates. If green (ACTIVATION) dominates, your activations are the bottleneck — consider gradient checkpointing or reducing batch size. If red (OPTIMIZER_STATE) dominates, you're using a stateful optimizer like Adam — consider switching to SGD.
- In a typical training run, you'll see green grow through the forward pass (activations accumulating), then shrink during the backward pass (activations freed as their gradient is computed), while orange (GRADIENT) grows during backward, and red (OPTIMIZER_STATE) appears at the optimizer step.

---

## 7. Known Limitations

**Granularity:** We hook at the `nn.Module` level, not the autograd Function level. One `nn.Linear` module encompasses multiple autograd operations (matmul, bias add). For a finer-grained graph, you'd need to hook into `torch.autograd.Function` directly or use `torch.fx` tracing.

**CPU memory accuracy:** `tracemalloc` doesn't capture PyTorch's C++ allocator, so CPU memory deltas are always 0. The static timeline (using formula-based `nbytes`) is the correct tool for CPU analysis.

**One iteration only:** The profiler is designed for a single forward+backward+step. Gradient accumulation across multiple batches, learning rate schedulers, and multi-iteration artifacts are not modeled.

**In-place operations:** Operations like `tensor.add_(other)` modify a tensor in-place. PyTorch's autograd requires that tensors saved for backward are not modified in-place after being saved. If your model uses in-place ops on leaf tensors, the profiler's tensor identity tracking may be incorrect.

**Dynamic control flow:** If your model's `forward()` has `if`/`for` statements that depend on the input data, the computational graph can change between iterations. This profiler captures the graph for one specific input.

---

## 8. Test Suite

The test suite lives in `tests/test_profiler.py` and contains **100 tests** runnable with:
```bash
python -m pytest tests/
# or
python -m unittest tests.test_profiler -v
```

### 8.1 Testing philosophy: what could go wrong?

Before writing a single test, we performed a systematic audit of every assumption embedded in the codebase. The key question for each component: *what property does this code rely on that, if violated, would silently corrupt results?* Silent corruption is the worst kind of bug in a profiler — the code runs fine but the numbers are wrong.

The 100 tests are organized into nine classes.

The failure modes fell into several families:

**Tensor identity failures.** `force_create` must always produce a new `tensor_id` even when called twice on the same tensor. `get_or_create` must be idempotent. A tensor at a known parameter address must always be classified as PARAMETER, regardless of any hint passed by the caller. These can fail silently: the graph builds, the chart renders, but some activations are silently merged with others or a gradient is classified as a parameter.

**Memory-reuse misclassification.** After a tensor is freed, the allocator may give the same `(data_ptr, shape)` to the next allocation. If a gradient is born at the address of a freed activation, `get_or_create` must detect the role mismatch and create a fresh entry rather than returning the stale activation. If `mark_optimizer_state` finds a stale gradient at the same address, it must not mutate that gradient's record — it must create a new one. Both failures produce wrong role counts in the output.

**Topological sort violations.** All FORWARD nodes must precede all BACKWARD nodes, which must precede the OPTIMIZER node. Any edge in the graph must point from an earlier step to a later one (DAG property). Node and tensor IDs must be globally unique. These can fail when hook firing order is disrupted, e.g., by hooking container modules.

**Lifetime underestimation.** After `compute_lifetimes()`, no activation may have a `None` lifetime. `first_use_op` must be ≤ `last_use_op` everywhere. Most critically, activations must have their lifetimes extended to the backward step of their producing module — without Pass 3, the peak memory estimate is wrong for every architecture.

**Categorization counts.** The number of PARAMETER tensors must exactly match `len(list(model.parameters()))`. Every parameter must have a gradient of matching shape. Adam produces 3 optimizer states per parameter, SGD without momentum produces 0, SGD with momentum produces 1. These are testable arithmetic invariants that catch a whole class of registration bugs.

**API contract.** Calling `run()` twice, or calling `get_report()`/`visualize()` before `run()`, must raise `RuntimeError`. These guard against state corruption when the profiler is used incorrectly.

**Visualizer soundness.** The static timeline must never contain negative values (no step can "un-allocate" more than was allocated). Activations must still be nonzero during the backward pass (confirming Pass 3 actually extended their lifetimes). The output PNG must be created and non-empty.

### 8.2 Test categories

#### `TestTensorRegistry` — 19 tests (unit tests, no full profiler run)

These test the registry in isolation by directly calling its methods on hand-constructed tensors.

| Test | What it catches |
|---|---|
| `test_force_create_gives_distinct_ids_same_tensor` | `_next_id` must increment on every call, even for the same tensor object |
| `test_force_create_gives_distinct_ids_same_ptr_shape` | Storage aliases (views of the same buffer) must still get distinct IDs |
| `test_force_create_overwrites_ptr_shape_mapping` | The latest `force_create` wins in the `_ptr_shape_to_meta` dict |
| `test_get_or_create_same_tensor_returns_same_id` | Standard idempotency — same tensor, same result |
| `test_get_or_create_role_preserved_on_second_call` | Classification is sticky; a second lookup doesn't reclassify |
| `test_gradient_hint_on_activation_address_creates_fresh_entry` | Memory-reuse guard: GRADIENT claiming freed ACTIVATION address → new entry |
| `test_optimizer_state_hint_on_activation_address_creates_fresh_entry` | Same guard for OPTIMIZER_STATE |
| `test_mark_optimizer_state_overwrites_gradient_entry` | The *old* gradient entry in `_registry` must stay as GRADIENT (not mutated) |
| `test_mark_gradient_does_not_mutate_activation_at_same_address` | Regression: `mark_gradient` must not corrupt a live activation at the same address |
| `test_parameter_wins_over_gradient_hint` | PARAMETER classification is unconditional for known parameter data_ptrs |
| `test_parameter_wins_over_activation_hint` | Same, with ACTIVATION hint |
| `test_mark_gradient_upgrades_to_longer_name` | `"grad_weight"` → `"grad_fc1.weight"` upgrade |
| `test_mark_gradient_keeps_longer_existing_name` | Long existing name is not downgraded to a short new one |
| `test_all_by_role_returns_correct_subset` | Role filter returns exactly the tensors with the requested role |
| `test_classify_activation_by_grad_fn` | `grad_fn is not None` → ACTIVATION |
| `test_classify_other_no_grad_fn_no_param` | Plain tensor, no grad_fn, not a param → OTHER |
| `test_all_tensor_ids_unique_in_registry` | 20 rapid `force_create` calls → all distinct IDs |
| `test_force_create_old_entry_survives_in_registry` | Old `TensorMeta` stays in `_registry` after `force_create` overwrites the ptr→meta mapping |
| `test_mark_gradient_equal_length_name_uses_new` | Equal-length names: the newer (more qualified) name wins |

#### `TestGraphTopoSort` — 13 tests

These run a full profiling iteration on the 2-layer MLP and inspect the graph structure.

| Test | What it catches |
|---|---|
| `test_all_forward_before_backward` | Max FORWARD index < min BACKWARD index |
| `test_all_backward_before_optimizer` | Max BACKWARD index < OPTIMIZER index |
| `test_exactly_one_optimizer_node` | Exactly one OPTIMIZER node per run |
| `test_node_count_two_layer_mlp` | 4 leaves × 2 (fwd+bwd) + 1 optimizer = 9 nodes |
| `test_node_count_single_linear` | 1 leaf → 3 nodes |
| `test_all_node_ids_unique` | No duplicate node_ids |
| `test_forward_node_count_equals_leaf_count` | Forward nodes = leaf modules |
| `test_backward_node_count_equals_leaf_count` | Backward nodes = leaf modules |
| `test_topo_order_respects_edges` | Every edge src appears before dst in topo order |
| `test_frozen_graph_rejects_new_nodes` | `freeze()` enforces immutability |
| `test_frozen_graph_rejects_new_edges` | Same for edges |
| `test_add_node_duplicate_id_overwrites_silently` | Inserting a node with an existing `node_id` replaces the old one |
| `test_topo_order_stable_across_calls` | `nodes_in_topo_order()` returns the same sequence on repeated calls |

#### `TestComputeLifetimes` — 13 tests

These are the most important correctness tests. They verify that Pass 3 of `compute_lifetimes()` correctly extends activation lifetimes to match what PyTorch's autograd actually keeps alive.

| Test | What it catches |
|---|---|
| `test_no_activation_has_none_lifetime` | All three passes completed without leaving any None |
| `test_first_leq_last_for_all_tensors` | Monotonicity: `first_use_op ≤ last_use_op` everywhere |
| `test_sigmoid_output_extends_to_sigmoid_backward` | Sigmoid saves its output: `sig1.output.last = sig1.backward step` |
| `test_relu_output_extends_to_relu_backward` | ReLU backward needs output as a mask |
| `test_tanh_output_extends_to_tanh_backward` | Tanh backward needs output to compute 1 - tanh²(x) |
| `test_linear_input_activation_extends_to_linear_backward` | Linear backward needs its input activation (dL/dW = dL/dZ^T @ input) |
| `test_final_layer_output_fixup_last_equals_first` | Output with no hook-visible consumer gets fixup: last = first |
| `test_deep_uniform_network_all_activations_distinct` | 5-layer same-shape net: ≥10 distinct activations (tests `force_create`) |
| `test_dropout_output_extends_to_dropout_backward` | Dropout saves its mask (same shape as output) |
| `test_activation_live_ops_positive` | Every activation has a live span of at least 1 step |
| `test_activations_have_first_use_op_set` | Pass 1 reached every activation |
| `test_compute_lifetimes_forward_only_no_backward` | Graph with no backward nodes: lifetimes still complete without error |
| `test_lifetime_indices_within_node_count` | All `first_use_op` and `last_use_op` values fall within `[0, n_steps - 1]` |

#### `TestProfilerCategorization` — 12 tests

These run the profiler on various model/optimizer combinations and assert exact counts and shapes.

| Test | What it catches |
|---|---|
| `test_parameter_count_no_bias` | PARAMETER count matches `len(list(model.parameters()))` |
| `test_parameter_count_with_bias` | Bias parameters are tracked (not just weights) |
| `test_every_parameter_has_a_gradient` | Every param shape appears in the gradient set |
| `test_activation_count_two_layer_mlp` | Exactly 4 ACTIVATION tensors for 4 leaf modules |
| `test_activation_names_match_modules` | `fc1.output`, `sig1.output`, etc. are present |
| `test_adam_optimizer_state_count` | Adam: 3 × num_params states (step, exp_avg, exp_avg_sq) |
| `test_sgd_no_momentum_zero_optimizer_states` | SGD without momentum: 0 states |
| `test_sgd_with_momentum_one_state_per_param` | SGD+momentum: 1 × num_params states |
| `test_adamw_optimizer_state_count` | AdamW: 3 × num_params (same structure as Adam) |
| `test_bias_parameters_tracked_with_gradients` | Bias params and their gradients both appear |
| `test_input_x_classified_as_other` | X, Y, L are OTHER |
| `test_rmsprop_state_count` | Version-robust: introspects the actual optimizer state keys |

#### `TestGraphIntegrity` — 8 tests

These verify the structural soundness of the graph — edges point to real nodes, tensor IDs exist in the registry, etc.

| Test | What it catches |
|---|---|
| `test_no_self_loops_in_edges` | An operation cannot depend on its own output |
| `test_edges_reference_valid_nodes` | No dangling edge endpoints |
| `test_edges_reference_valid_tensor_ids` | No edge references a tensor not in the registry |
| `test_all_nodes_have_profile` | No node was built without timing/memory data |
| `test_all_profiles_have_nonnegative_wall_time` | Timer was called correctly |
| `test_output_tensor_ids_appear_in_registry` | Every output tensor is registered |
| `test_input_tensor_ids_appear_in_registry` | Every input tensor is registered |
| `test_all_node_phases_valid` | All phases are valid OpPhase enum members |

#### `TestAPIGuardrails` — 10 tests

These test the public API contract: what happens when the user calls methods in the wrong order or with unusual inputs.

| Test | What it catches |
|---|---|
| `test_second_run_raises_runtime_error` | `_ran` flag prevents double execution |
| `test_get_report_before_run_raises` | Cannot query before running |
| `test_visualize_before_run_raises` | Cannot visualize before running |
| `test_batch_size_one` | Minimum batch size works; no shape-dependent crashes |
| `test_get_report_returns_profile_report` | Return type is correct |
| `test_large_batch_runs_correctly` | Batch of 256: still exactly 4 activations tracked |
| `test_wrapped_optimizer_zero_grad_delegates` | `zero_grad()` delegation clears all parameter gradients |
| `test_wrapped_optimizer_param_groups_accessible` | `param_groups` property delegates to wrapped optimizer |
| `test_no_param_model_without_device_raises` | Model with no parameters and no explicit device raises cleanly |
| `test_print_summary_does_not_crash` | `report.print_summary()` completes without exception on a valid report |

#### `TestArchitectures` — 9 tests

These profile diverse architectures to catch bugs that only appear in specific model shapes.

| Test | What it catches |
|---|---|
| `test_single_linear_no_bias` | Minimum viable model: 1 param, 1 activation, 3 nodes |
| `test_three_layer_relu_net` | 5 activations, 3 parameters, correct counts |
| `test_network_with_bias_all_roles_correct` | Bias params have gradients of matching shape |
| `test_dropout_tracked_correctly` | Dropout leaf module appears in forward and backward |
| `test_deep_net_correct_activation_count` | Non-uniform 5-layer net: 5 distinct activations |
| `test_tanh_network_lifetimes_complete` | All lifetimes non-None, monotone |
| `test_mixed_activations_relu_sigmoid` | Alternating activation types: 5 activations tracked |
| `test_deep_relu_mlp_node_and_tensor_counts` | 4-hidden-layer ReLU MLP: 19 nodes, 9 activations, 5 params, 15 Adam states |
| `test_deep_relu_relu_activations_have_extended_lifetimes` | ReLU outputs have `last_use_op > first_use_op` (Pass 3 fired) |

#### `TestMemoryVisualizer` — 9 tests

| Test | What it catches |
|---|---|
| `test_static_timeline_creates_file` | PNG is created and non-empty |
| `test_dynamic_timeline_creates_file` | Dynamic mode also works |
| `test_invalid_mode_raises` | Unknown mode string raises ValueError |
| `test_static_timeline_nonnegative_values` | No step has negative memory (per role) |
| `test_static_timeline_activation_nonzero_during_backward` | Pass 3 extended lifetimes into the backward phase |
| `test_static_peak_is_nonnegative` | Total peak ≥ 0 |
| `test_static_timeline_length_equals_node_count` | Timeline array matches number of graph nodes |
| `test_dynamic_timeline_length_equals_node_count` | Dynamic mode timeline length also matches node count |
| `test_static_activation_nonzero_at_first_use_op` | Each activation contributes nonzero memory at its `first_use_op` step |

#### `TestPass4Lifetimes` — 7 tests

These verify the Pass 4 normalization of persistent tensor lifetimes, which ensures the static memory chart shows correct per-role contributions across the full timeline.

| Test | What it catches |
|---|---|
| `test_parameters_first_use_op_is_zero` | Every PARAMETER has `first_use_op == 0` after Pass 4 |
| `test_parameters_last_use_op_is_final_step` | Every PARAMETER has `last_use_op == n_steps - 1` |
| `test_optimizer_states_persist_to_final_step` | Every OPTIMIZER_STATE has `last_use_op == n_steps - 1` |
| `test_param_grad_memory_nonzero_at_optimizer_step` | Terminal gradients are live at the optimizer step (chart shows orange at that bar) |
| `test_intermediate_gradients_not_extended_to_optimizer_step` | Flow gradients (`dL/dZ`) that are consumed by later backward nodes are NOT extended |
| `test_parameter_memory_nonzero_at_step_zero` | PARAMETER contributes nonzero memory at step 0 (blue baseline starts at the beginning) |
| `test_optimizer_state_memory_at_final_step` | OPTIMIZER_STATE contributes nonzero memory at the final step |

### 8.3 Challenges discovered during testing

Writing the tests uncovered three additional issues that were invisible during manual validation. Two surfaced immediately when the tests were first run; the third was a flaky failure that only appeared when tests ran in sequence — a realistic condition the manual validation never exercised.

---

#### Challenge 9 (testing): Duplicate keyword argument in test helper

**Symptom:** `TypeError: got multiple values for keyword argument 'lr'` on the SGD tests.

**Root cause:** The shared `_profiler` helper in `TestProfilerCategorization` hardcoded `lr=1e-3`:
```python
opt = opt_cls(model.parameters(), lr=1e-3, **opt_kwargs)
```
The SGD-specific tests passed `lr=0.01` in `**opt_kwargs`. Python does not allow a keyword argument to appear both positionally and in `**kwargs`, so the call raised a `TypeError` before any optimizer was even constructed.

This error would not occur during normal use of the profiler — it was purely a test code issue. But it revealed a subtlety: test helpers that construct optimizers need to either let the caller fully control the learning rate or not hardcode it at all. A helper that says "I'll set sensible defaults but let the caller override" must use `setdefault`, not a positional keyword:

```python
# Wrong: caller cannot override lr
opt = opt_cls(model.parameters(), lr=1e-3, **opt_kwargs)

# Right: caller's lr wins if provided
opt_kwargs.setdefault("lr", 1e-3)
opt = opt_cls(model.parameters(), **opt_kwargs)
```

**Fix:** Changed the helper to call `opt_kwargs.setdefault("lr", 1e-3)` before unpacking `**opt_kwargs`.

---

#### Challenge 10 (profiler bug, found by tests): `mark_gradient` mutating a live activation's registry entry

**Symptom:** Flaky test failure — `test_three_layer_relu_net` failed roughly 1 in 14 runs with `AssertionError: 4 != 5` (wrong activation count). The failure was non-deterministic: it depended on the CPU allocator's free list state left behind by the 70+ tests that preceded it in the suite.

**Root cause:** The bug lived in `mark_gradient()`. When the executor calls `mark_gradient(param, param.grad, "fc1.weight")` after `loss.backward()`, the function looks up `param.grad`'s `(data_ptr, shape)` in `_ptr_shape_to_meta` and mutates whatever entry it finds:

```python
# BUGGY code
if key in self._ptr_shape_to_meta:
    meta = self._ptr_shape_to_meta[key]
    meta.role = TensorRole.GRADIENT   # ← unconditional mutation!
    ...
    return meta
```

In the 3-layer ReLU network, `relu2.output` and `grad_fc1.weight` both have shape `[8, 4]`. During the backward pass, relu2.output is eventually freed. The memory allocator then reuses that same address for `grad_fc1.weight`. At that point, `_ptr_shape_to_meta[(data_ptr, (8,4))]` still points to relu2.output's `TensorMeta`. When `mark_gradient` runs, it finds relu2.output's entry, mutates its role to `GRADIENT`, renames it to `"grad_fc1.weight"`, and returns it. relu2.output has been silently destroyed.

This is the same class of bug as Challenge 5 (SGD momentum buffer overwriting gradient entries), but it affects `mark_gradient` instead of `mark_optimizer_state`, and corrupts an *activation* record instead of a gradient record.

The bug only manifested when the allocator's free list was in a specific state — which happened non-deterministically depending on what memory earlier tests had allocated and freed. Running the test in isolation always passed because the allocator had a clean state.

**The test suite found it.** This is a concrete example of why running tests in sequence (not just in isolation) matters: the shared allocator state across tests creates realistic memory-reuse conditions that isolated tests cannot reproduce.

**Fix:** In `mark_gradient`, only mutate an entry when it's already a `GRADIENT`. If the existing entry is an ACTIVATION or OTHER, fall through to `get_or_create`, which handles the role mismatch via its memory-reuse guard (creating a fresh entry and leaving the old activation intact):

```python
def mark_gradient(self, param, grad, param_name):
    grad_name = f"grad_{param_name}"
    self._grad_ptrs[grad.data_ptr()] = param_name
    key = (grad.data_ptr(), tuple(grad.shape))
    if key in self._ptr_shape_to_meta:
        meta = self._ptr_shape_to_meta[key]
        if meta.role == TensorRole.GRADIENT:     # ← only update if it's already a gradient
            if len(grad_name) >= len(meta.name):
                meta.name = grad_name
            return meta
        # Role mismatch — fall through to create a fresh entry
    return self.get_or_create(grad, TensorRole.GRADIENT, grad_name)
```

A dedicated regression test (`test_mark_gradient_does_not_mutate_activation_at_same_address`) was added to the suite to prevent recurrence.

---

#### Challenge 11 (testing): RMSprop optimizer state count changed in PyTorch 2.x

**Symptom:** `AssertionError: RMSprop: expected 2 states, got 4`.

**Root cause:** Our test originally assumed RMSprop maintains one state tensor per parameter (`square_avg`). This was true in PyTorch 1.x. In PyTorch 2.0, the optimizer framework changed to store the iteration counter `step` as a *tensor* rather than a Python integer — making it a tracked state entry like any other. Default RMSprop (no momentum, not centered) now stores `step` + `square_avg` = 2 state tensors per parameter. Our hardcoded expected count of `n_params` (= 2) was off by a factor of 2 from the actual `2 * n_params` (= 4).

**Why this matters beyond the test:** This is actually an important real-world fact. If you switch from PyTorch 1.x to 2.x, RMSprop will use *twice* the optimizer state memory. Our profiler would correctly report the higher count in PyTorch 2.x, but a user comparing against PyTorch 1.x documentation would be surprised.

**Fix:** Made the test version-robust by introspecting the actual optimizer state after the first step, rather than asserting a hardcoded count:

```python
# Count actual per-param state tensors by inspecting the optimizer's state dict
actual_per_param = sum(
    1 for v in opt.state[next(iter(opt.state))].values()
    if isinstance(v, torch.Tensor)
)
expected = actual_per_param * n_params
self.assertEqual(len(opt_states), expected)
```

This test now works correctly regardless of PyTorch version, and simultaneously validates that the profiler accurately tracks *all* optimizer state tensors — not just the ones we expected going in.

### 8.4 Running the tests

```bash
# From the cs265 directory
python -m unittest tests.test_profiler -v

# Or run a specific class
python -m unittest tests.test_profiler.TestComputeLifetimes -v

# Or a single test
python -m unittest tests.test_profiler.TestComputeLifetimes.test_sigmoid_output_extends_to_sigmoid_backward -v
```

Expected output: `100 passed in ~2s`.
