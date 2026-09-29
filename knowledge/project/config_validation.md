---
doc_id: project-config_validation
title: RunConfig validation
source: src/ml_experiment_agent/models/experiment.py
topic: config_validation
error_type: CONFIG_ERROR
---

# RunConfig validation

Config validation

RunConfig forbids unknown fields and validates positive learning_rate, batch_size, epochs and hidden_dim. precision is float32 only. ProjectSpec defines which config fields may be modified. Unknown or invalid parameters must be rejected before submission; they must not be silently renamed or clamped. A runtime dimension error is distinct from a schema validation error. Diagnostics explain the observed failure without writing a patch.

