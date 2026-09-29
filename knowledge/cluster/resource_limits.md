---
doc_id: cluster-resource_limits
title: Project resource policy
source: configs/policy.yaml
topic: resource_policy
error_type: RESOURCE_ERROR
---

# Project resource policy

Resource limits

Default project limits are GPU <= 1, CPU <= 8, memory <= 32 GB, timeout <= 60 minutes and at most 3 parallel jobs. Policy rejection happens before backend submission. Paths must resolve under allowed_root and entrypoints must match the registered ProjectSpec. A request requiring approval is BLOCKED rather than silently allowed. This policy is validation of requests, not an operating-system sandbox or proof of actual peak usage.

