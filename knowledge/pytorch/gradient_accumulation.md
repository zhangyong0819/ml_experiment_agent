---
doc_id: pytorch-gradient_accumulation
title: Gradient accumulation context
source: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
topic: gradient_accumulation
error_type: CUDA_OOM
---

# Gradient accumulation context

Gradient accumulation

Several microbatch backward passes can contribute to one optimizer step. Effective batch size depends on microbatch size and accumulation steps. Accumulation does not remove memory used by weights, optimizer state or a single oversized microbatch. The toy project implements gradient_accumulation_steps and handles the last partial group. Read the config rather than assuming accumulation is enabled. Diagnose only; changing these settings belongs to a later recovery phase.

