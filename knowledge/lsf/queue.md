---
doc_id: lsf-queue
title: Queue and resource availability
source: https://www.ibm.com/docs/en/spectrum-lsf
topic: lsf_queue
error_type: RESOURCE_ERROR
---

# Queue and resource availability

Queue state

QUEUED means a submission is waiting and RUNNING means execution has started. Resource availability or queue policy can delay execution. A bounded graph monitoring stop is not evidence that the backend job itself timed out or failed. The project preserves the last known Attempt status when monitoring stops. Diagnose scheduler errors only from recorded evidence; do not fabricate queue names or cluster resource usage.

