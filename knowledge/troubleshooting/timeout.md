---
doc_id: troubleshooting-timeout
title: Timeout versus monitor bound
source: src/ml_experiment_agent/graph/nodes/monitoring.py
topic: timeout
error_type: TIMEOUT
---

# Timeout versus monitor bound

Timeout evidence

A backend TIMED_OUT event and a graph monitor-cycle limit are different. Explicit scheduler TIME LIMIT or a subprocess TimeoutExpired supports a timeout diagnosis. Reaching max_monitor_cycles only stops this workflow's polling; the job can remain QUEUED/RUNNING. Record the applicable limit and observed state when known. The agent does not cancel an active job or increase a timeout.

