# Resume preparation notes

- Goal: stateful experiment planning, controlled execution, diagnosis and bounded recovery.
- Stack: Python 3.11, Pydantic, SQLAlchemy/SQLite, PyTorch/scikit-learn, LangGraph, LangChain tools/ChatDeepSeek, multilingual E5/FAISS and FastAPI.
- Design: authoritative application DB, separate checkpoints, append-only attempts, read-only tools, deterministic policy, human approval for scientific changes.
- LocalBackend uses real Popen CPU execution. A labelled OOM-shaped failure fixture triggers recovery and actual successful training; no real CUDA OOM is caused.
- Manual HPC Bridge provides validated export/import and recovery with an explicit human execution boundary. Offline fixtures do not prove real cluster execution.
- Acceptance: 321 tests; FastAPI locally verified as a single-instance service.
- Offline evaluation: 37 cases, task success 35/37 (94.6%), diagnosis 14/14, fake RAG Hit@3 5/7, recovery 5/5, average retry count 1.0, unsafe blocking 9/9, false blocking 0/7.
- Real E5 previously retrieved all three tested cluster targets at Top-1. RAG on/off diagnosis was 6/6 in both settings; no diagnosis accuracy improvement is demonstrated.
- Real DeepSeek and automated real LSF execution remain unverified. No production cluster, large-model training or distributed deployment claim is warranted.
- School/lab document redistribution permission remains unresolved and blocks publication.

Suggested bullets:

1. Built a LangGraph experiment workflow with SQLite checkpoints and persistent Run/Attempt history across controlled retries and human approval, covered by 321 offline tests.
2. Combined read-only tool calling and documentation RAG for structured diagnosis; deterministic policy validates proposed fixes before execution. Evaluated 37 fixtures with 35/37 task success and 9/9 unsafe actions blocked.
3. Implemented real local process execution and a manual HITL HPC export/import boundary, alongside a single-instance FastAPI service verified through local HTTP.
