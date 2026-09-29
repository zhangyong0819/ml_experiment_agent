---
doc_id: project-toy_usage
title: Toy classification execution
source: toy_project/train.py
topic: toy_project
error_type: UNKNOWN
---

# Toy classification execution

Toy project usage

The registered toy_classification project creates fixed local make_classification data and a small CPU MLP. Train reads YAML and writes checkpoint.pt and metadata.json; evaluate writes accuracy, macro_f1 and ece to metrics.json. No dataset download or GPU is required. MockBackend simulates scheduling independently of real CPU artifact preparation. An OOM mock log is a deliberate fixture, not evidence that this CPU machine has a failing GPU.

