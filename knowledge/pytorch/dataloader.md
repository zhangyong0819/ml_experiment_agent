---
doc_id: pytorch-dataloader
title: DataLoader workers
source: https://docs.pytorch.org/docs/stable/data.html
topic: dataloader
error_type: RESOURCE_ERROR
---

# DataLoader workers

DataLoader worker resources

num_workers controls loader subprocesses. The project reserves CPUs for the training process plus its data workers. More workers may increase process and host-memory overhead; CUDA OOM and host resource exhaustion are different observations. For worker exits, collect the traceback, worker count and available scheduler metadata. Do not infer an ImportError, filesystem failure or memory kill from a generic worker-exited message alone.

