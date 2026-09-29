---
doc_id: cluster-preemption
title: Preemption is not a model bug
source: https://www.ibm.com/docs/en/spectrum-lsf
topic: preemption
error_type: PREEMPTED
---

# Preemption is not a model bug

Scheduler preemption

Preemption means scheduler resources were reclaimed under cluster policy. The mock scenario logs PREEMPTED and records ErrorType.PREEMPTED. It does not demonstrate a defect in the model or configuration. Check whether a checkpoint artifact actually exists before mentioning possible continuation. retriable in a diagnosis is only an advisory flag. No automatic resubmission, checkpoint restoration or retry occurs in Phase 3A.

