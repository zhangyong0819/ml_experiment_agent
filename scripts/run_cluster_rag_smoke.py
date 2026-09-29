"""Real cached multilingual E5 checks against user-supplied cluster documents."""
from pathlib import Path
import json
import sys
from ml_experiment_agent.rag.documentation import DocumentationIndex
from ml_experiment_agent.rag.embeddings import e5_embeddings, MODEL_NAME


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    index = DocumentationIndex.load(root / "data/faiss", e5_embeddings(root / "data/hf_cache", local_files_only=True), MODEL_NAME)
    queries = [
        ("组里有哪些可用的 LSF 队列？", "group_lsf_queues", ["734090d!", "944090d!", "9654p6000ib!"]),
        ("LSF 如何使用 bsub 命令行和脚本提交作业？", "cluster-submit_jobs", ["bsub"]),
        ("LSF bjobs -l JOBID 查看未结束作业详情，bpeek 查看 stdout stderr", "cluster-control_jobs", ["bjobs", "bpeek"]),
    ]
    observations = []
    for query, expected_id, text in queries:
        results = index.retrieve(query, 3)
        hits = [r for r in results if r["doc_id"] == expected_id]
        passed = bool(hits) and all(t in "\n".join(r["snippet"] for r in hits) for t in text)
        observations.append({"query": query, "expected_doc_id": expected_id, "passed": passed,
            "retrieved": [{"doc_id": r["doc_id"], "score": r["score"]} for r in results]})
    path = root / "data/cluster_rag_smoke.json"
    path.write_text(json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(observations, ensure_ascii=False, indent=2))
    if not all(o["passed"] for o in observations):
        raise RuntimeError("Cluster RAG smoke failed; see actual retrieval results")
    print("CLUSTER_RAG_SMOKE=PASS (real multilingual E5, top 3)")


if __name__ == "__main__":
    main()
