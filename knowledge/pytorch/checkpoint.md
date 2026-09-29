---
doc_id: pytorch-checkpoint
title: Checkpoint discovery and evaluation
source: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
topic: checkpoint
error_type: PATH_ERROR
---

# Checkpoint discovery and evaluation

Checkpoint artifacts

The toy project saves checkpoint.pt with model state_dict and hidden_dim. Evaluation loads an existing checkpoint on CPU using weights_only=True. A successful mock scheduler status alone does not create model weights; the graph's explicit toy artifact preparation does. Discovery requires checkpoint, config and logs. A missing checkpoint is an artifact failure, not successful evaluation. Diagnostics must not create a replacement file or retrain the model.

