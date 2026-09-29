---
doc_id: project-artifacts
title: Artifact and database authority
source: src/ml_experiment_agent/services/artifact_service.py
topic: artifacts
error_type: PATH_ERROR
---

# Artifact and database authority

Artifact authority

Application SQLite is the source of truth for experiments, runs, attempts and artifact references. The existence of an Artifact row does not guarantee that the file still exists, so discovery checks the resolved path. Files must remain under the attempt output directory and allowed_root. The diagnosis tool lists metadata and existence only; it does not read checkpoints or arbitrary files. A checkpoint resume must not silently recreate deleted training output.

