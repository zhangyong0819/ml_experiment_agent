import csv
import json
from pathlib import Path


def format_metric(name: str, metric: dict) -> str:
    if metric["value"] is None:
        return "N/A (n=0)"
    value = f"{metric['value']:.3f}" if name == "Average Retry Count" else f"{metric['value']:.1%}"
    return f"{value} (n={metric['denominator']})"


def export_results(report: dict, output: Path) -> None:
    """Write plain, inspectable JSON/CSV/Markdown with explicit denominators and scope."""
    output.mkdir(parents=True, exist_ok=True)
    payload = report | {key: [r.model_dump(mode="json") for r in report[key]] for key in ("results", "comparison_results")}
    (output / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = payload["results"] + payload["comparison_results"]
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    lines = ["# Agent Engineering Evaluation", "", f"Mode: **{report['mode']}**; embedding: `{report['embedding_id']}`; cases: {report['case_count']}.", "",
        "Task-specific fixtures, not a general agent benchmark. Default models and token-hash embeddings are fake; "
        "retrieval still runs actual FAISS over the current documents. Successful graph cases run real CPU toy training/evaluation.", "",
        "Task success means all case expectations passed, including correctly stopping unsafe/unrecoverable work. "
        "It does not mean every training job succeeded. No LLM judge is used.", "",
        "## Overall", "", "| Metric | Result |", "| --- | ---: |"]
    lines += [f"| {name} | {format_metric(name, metric)} |" for name, metric in report["overall"].items()]
    for title, key in (("RAG comparison", "rag_comparison"), ("Recovery comparison", "recovery_comparison")):
        variants = report[key]
        lines += ["", f"## {title}", "", "| Metric | " + " | ".join(variants) + " |", "| --- | " + " | ".join("---:" for _ in variants) + " |"]
        names = ("Diagnosis Accuracy", "RAG Hit@3") if key == "rag_comparison" else ("Task Success Rate", "Recovery Success Rate", "Average Retry Count")
        lines += ["| " + name + " | " + " | ".join(format_metric(name, metrics[name]) for metrics in variants.values()) + " |" for name in names]
    lines += ["", "Fake diagnosis recognizes log signatures. Equal diagnosis accuracy with/without RAG is expected and is not evidence of improved reasoning. "
              "RAG Hit@3 measures actual evidence retrieval. Recovery comparison uses exactly the same cases and graph with RecoveryService omitted in diagnosis-only mode.",
              "", "## Category breakdown", "", "| Category | Task Success |", "| --- | ---: |"]
    lines += [f"| {category} | {format_metric('Task Success Rate', metrics['Task Success Rate'])} |" for category, metrics in report["categories"].items()]
    lines += ["", "## Representative failures", ""]
    failures = [r for r in report["results"] if not r.success]
    lines += [f"- `{r.case_id}`: {r.notes}" for r in failures] or ["None in this fixture set; this is not a claim of general reliability."]
    lines += ["", "## Scope and denominators", "",
        "- Diagnosis accuracy: only cases with an expected error type; execution errors count as incorrect.",
        "- RAG Hit@3: cases with expected document IDs/topics; any matching ID OR topic within one query's top 3 is a hit. Required snippets are checked separately for task success.",
        "- Recovery success / average retries: only cases explicitly marked recoverable; ImportError STOP and exhaustion cases are excluded.",
        "- Unsafe blocking and false blocking use separate declared unsafe/legal sets; both are always reported.",
        "- Required Tool Recall is per-case recall averaged over cases declaring tools; extra read-only tools do not hurt the score.",
        "- Empty denominators are N/A, never zero or a claimed perfect score.",
        "- Fake runs omit latency/timestamps/random identifiers so JSON/CSV/Markdown can be reproduced. Real runs may include observed latency/token usage.",
        "- Cluster manuals are knowledge only. No shell, LSF submission, container runtime, or remote login is executed."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
