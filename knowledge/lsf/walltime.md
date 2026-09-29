---
doc_id: lsf-walltime
title: LSF walltime termination
source: https://www.ibm.com/docs/en/spectrum-lsf
topic: lsf_walltime
error_type: TIMEOUT
---

# LSF walltime termination

LSF walltime / timeout

A job can be terminated after its walltime limit. Look for explicit scheduler time-limit evidence and compare it with the requested timeout_minutes. A long queue wait is not necessarily a training timeout; queued and running states differ. The mock TIMEOUT scenario emits TIME LIMIT and uses TIMED_OUT. The agent records the evidence and does not extend walltime, resubmit the job or claim scheduler access.

