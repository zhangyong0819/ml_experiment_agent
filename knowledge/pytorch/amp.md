---
doc_id: pytorch-amp
title: Precision and AMP boundaries
source: https://docs.pytorch.org/docs/stable/amp.html
topic: precision
error_type: CONFIG_ERROR
---

# Precision and AMP boundaries

AMP and precision

Automatic mixed precision uses selected lower-precision operations and may change memory usage. It is not enabled merely by requesting a GPU. This registered toy project supports float32 only. An fp16/bf16 configuration is rejected by RunConfig, even if a library supports AMP elsewhere. A diagnosis should distinguish a schema rejection from an actual runtime CUDA OOM. Do not suggest that AMP has been used when the config or logs do not show it.

