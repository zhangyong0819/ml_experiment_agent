---
doc_id: lsf-memory
title: LSF memory requests and limits
source: https://www.ibm.com/docs/en/spectrum-lsf
topic: lsf_memory
error_type: RESOURCE_ERROR
---

# LSF memory requests and limits

LSF memory context

Scheduler memory requests, enforced memory limits and GPU VRAM are separate concepts. Units and enforcement depend on cluster configuration; inspect the local project/cluster documentation and scheduler log before interpreting numbers. Host-memory limit termination is not automatically CUDA_OOM. Record the actual termination reason and requested memory if available. No LSF backend or scheduler command tool is implemented in this project phase.

