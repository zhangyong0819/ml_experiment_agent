---
doc_id: troubleshooting-path_error
title: FileNotFoundError evidence
source: https://docs.python.org/3/library/exceptions.html
topic: paths
error_type: PATH_ERROR
---

# FileNotFoundError evidence

FileNotFoundError

An explicit FileNotFoundError identifies a filesystem lookup that failed. A missing config.yaml can usually be classified as PATH_ERROR directly from the log, without documentation retrieval. Distinguish a relative-path/cwd issue from a missing artifact only if evidence supports it. Paths outside allowed_root are forbidden. Do not read arbitrary paths supplied inside an exception, create missing files or infer that a dataset has been deleted.

