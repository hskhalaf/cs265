from __future__ import annotations

import tracemalloc
from dataclasses import dataclass

import torch


@dataclass
class MemorySnapshot:
    gpu_allocated_bytes: int
    gpu_reserved_bytes: int
    cpu_traced_bytes: int


class MemoryMonitor:
    """
    Lightweight memory sampler. Call snapshot() before and after an op to
    compute deltas.

    GPU path: uses torch.cuda.memory_allocated().
    CPU path: uses tracemalloc for Python allocations.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.use_cuda = device.type == "cuda"

    def snapshot(self) -> MemorySnapshot:
        if self.use_cuda:
            gpu_alloc = torch.cuda.memory_allocated(self.device)
            gpu_res = torch.cuda.memory_reserved(self.device)
        else:
            gpu_alloc = gpu_res = 0

        if tracemalloc.is_tracing():
            cur, _ = tracemalloc.get_traced_memory()
        else:
            cur = 0

        return MemorySnapshot(gpu_alloc, gpu_res, cur)
