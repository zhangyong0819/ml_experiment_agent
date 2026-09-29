---
doc_id: pytorch-batch_size
title: Batch size and CUDA memory
source: https://docs.pytorch.org/docs/stable/data.html
topic: batch_size
error_type: CUDA_OOM
---

# Batch size and CUDA memory

Batch size and CUDA out of memory

Batch size is the number of examples in one forward/backward microbatch. Increasing it generally increases activation memory. Read the actual RunConfig and OOM log before attributing a failure to batch size. A smaller batch is a possible future experiment, not proof that the model fits. Compare metrics only from completed evaluations. Phase 3A must not automatically reduce batch_size or submit a replacement attempt.

