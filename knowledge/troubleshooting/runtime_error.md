---
doc_id: troubleshooting-runtime_error
title: Unknown runtime failures
source: src/ml_experiment_agent/backends/mock.py
topic: runtime
error_type: RUNTIME_ERROR
---

# Unknown runtime failures

RuntimeError and insufficient evidence

A generic RuntimeError can arise from many causes. Inspect the exception text, run config and available logs. A non-finite loss observation does not prove a specific learning rate caused divergence. Use UNKNOWN with low confidence when the evidence is insufficient. Separate observed log fragments from model inference, and cite only documentation IDs actually returned by retrieval. Never manufacture traceback lines or success metrics.

