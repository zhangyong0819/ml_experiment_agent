# Publication review

Public release is blocked until redistribution permission is confirmed for the nine documents labelled `university_cluster_manual` and `knowledge/cluster/group_lsf_queues.md` (`lab_internal_note`). Their inclusion in this local copy is not publication authorization. Do not upload this repository yet.

School-manual example account names, home paths and example passwords were replaced with placeholders; public service addresses were retained. These substitutions do not resolve redistribution permission. No RAG index was regenerated.

No LICENSE found. No license was selected automatically.

The original development directory is retained separately. This copy excludes its environment file, virtual environment, model caches, databases and execution artifacts. Outputs generated during acceptance remain ignored under `data/`.

## Final acceptance

- Complete pytest after cleanup: 321 passed in 165.64 seconds.
- Local demo: real CPU subprocess success and labelled failure -> recovery -> second Attempt success.
- HPC Bridge demo: offline fixture export -> failed import -> diagnosis/recovery -> new bundle -> successful import -> report.
- Actual Uvicorn HTTP GET /health: 200, mock backend.
- No credentials or environment file copied. No target files exceeded 10, 50 or 100 MiB at inspection; generated acceptance artifacts are ignored.
- Thirty knowledge documents have unique IDs. Example personal paths/accounts were replaced; public service addresses remain.
- Source, tests, scripts, project specifications and configuration match the original copy; dependencies were not upgraded.
- No Git repository/history exists in the target; no initialization or remote operations were performed.
- Tests used the existing development Python environment with imports explicitly pointing at this target copy. A fresh dependency installation was not repeated.

Repository is NOT ready for GitHub publication because: school/lab knowledge document redistribution permission remains unconfirmed.
