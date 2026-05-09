# Phase 3 Plan: FX Graph Rewriting for Activation Checkpointing

Phase 2 chooses which activations should be recomputed.  Phase 3 makes that
decision real by editing the traced FX training graph so the selected
activations are not kept alive until backward.  After Phase 3, the AC memory
and latency plots can be measured directly instead of estimated by simulation.

## Goal

Given:

- a traced training-iteration `GraphModule`
- its `GraphProfiler`
- `SelectionResult.to_recompute`

produce a rewritten `GraphModule` that:

- recomputes each selected activation immediately before its first backward use
- redirects backward consumers to the recomputed tensor
- preserves the original forward, loss, backward, and optimizer semantics
- passes output/gradient correctness checks against the original graph

## Rewriting Algorithm

1. Build the retained boundary.

   Boundary nodes are tensors that should still be available during backward:
   placeholders, parameters, optimizer state, inputs, and retained activations.
   Recompute bodies stop when they reach this boundary.

2. For each selected activation in forward topological order:

   Find the first backward user of that activation.  This is the insertion
   point for the recomputation chain.

3. Gather the recomputation body.

   Walk backward from the selected activation through its input nodes until the
   retained boundary is reached.  The body contains every forward op needed to
   rebuild the activation, in topological order.  This should match
   `recompute_body_for_activation(...)` from Phase 2.

4. Copy the body into the backward region.

   Use `gm.graph.inserting_before(first_backward_user)` and
   `gm.graph.node_copy(...)`.  Maintain a map from original node to copied node
   so copied ops use copied inputs when needed and original boundary nodes when
   those inputs are retained.

5. Redirect backward consumers.

   For each backward user of the original activation, replace the original
   activation input with the recomputed copy.  Forward/loss users must keep
   using the original tensor.

6. Clean and recompile.

   Run `gm.graph.lint()` and `gm.recompile()`.  Remove dead detach/view helper
   nodes only if they have no users.

## Correctness Checks

Run these before trusting any measured AC numbers:

- fixed seed, same model, same inputs
- original graph output vs rewritten graph output: max abs diff <= `1e-4`
- original parameter gradients vs rewritten parameter gradients: max abs diff <= `1e-4`
- no selected activation remains live until its original backward users in the
  rewritten graph
- run at least `dummy`, `resnet18`, and one BERT batch

## Measurement Changes

After rewriting works, add a measured Phase 3 path:

- trace and profile the original graph
- run Phase 2 selection
- rewrite the graph
- profile the rewritten graph with `GraphProfiler`
- plot measured `no_ac` vs measured `ac`

At that point:

- `phase2_peak_vs_batch_*` remains the selector/simulator result
- Phase 3 plots should use names like `phase3_peak_vs_batch_*`
- labels should change from `AC estimate` to `AC measured`

## Main Risks

- Aliasing: views and getitem nodes must not create double-counted storage or
  broken input replacement.
- Multiple backward consumers: all backward users after the insertion point must
  consume the recomputed tensor.
- Shared recomputation: the Phase 2 cost model assumes independent recompute
  chains.  Phase 3 should initially match that exactly before trying to share
  copied subgraphs.
- In-place optimizer ops: the rewrite should only touch forward activations and
  backward consumers, never optimizer state updates.
