---
doc_id: pytorch-cuda_oom
title: CUDA out of memory
source: https://docs.pytorch.org/docs/stable/notes/cuda.html
topic: cuda_oom
error_type: CUDA_OOM
---

# CUDA out of memory

CUDA out of memory / 显存不足

Observed torch.cuda.OutOfMemoryError means a CUDA allocation failed. Capture the requested allocation, device capacity and current batch size from the real log; do not invent GPU measurements. Larger batches and activations can consume more memory. Batch size, sequence length and model dimensions are relevant context. Gradient accumulation can change the effective batch while using smaller microbatches, but does not guarantee recovery. Phase 3A only records this diagnosis; it never changes a configuration or retries.

