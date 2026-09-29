import json
from pathlib import Path
from ..db.models import Experiment, Diagnosis, Attempt, Run
from ..db.repository import Repository
from ..models.comparison import ExperimentComparison
from .policy_engine import PolicyEngine


def cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


class ReportService:
    """Stable Markdown from application facts; atomically replace one report per experiment."""

    def __init__(self, repository: Repository, policy: PolicyEngine):
        self.repo, self.policy = repository, policy

    def render(self, comparison: ExperimentComparison, stop_reason: str | None = None) -> str:
        experiment = self.repo.get(Experiment, comparison.experiment_id)
        names = {r.run_id: r.name for r in comparison.runs}
        lines = ["# ML Experiment Report", "", "## Experiment Goal", "", experiment.goal, "",
                 f"Experiment: `{experiment.id}`", "", "## Run Summary", "",
                 "Complete result." if comparison.status == "COMPLETE" and not stop_reason else
                 "**Partial result**: failed, blocked, incomplete, or unevaluated runs are not treated as successful.",
                 f"Comparison status: {comparison.status}", "", "## Config Differences", ""]
        lines.append("The table below describes the original experimental design. Attempt-specific retry changes are recorded separately in Recovery History.")
        keys = sorted({k for r in comparison.runs for k in r.config} - {"output_dir"})
        differing = [k for k in keys if len({json.dumps(r.config.get(k), sort_keys=True) for r in comparison.runs}) > 1]
        if differing:
            lines += ["| Run | " + " | ".join(differing) + " |", "| --- | " + " | ".join("---" for _ in differing) + " |"]
            lines += ["| " + cell(r.name) + " | " + " | ".join(cell(r.config.get(k)) for k in differing) + " |" for r in comparison.runs]
        else:
            lines.append("No differing configuration fields (attempt output directories excluded).")
        lines += ["", "## Execution Status", "", "| Run | Run status | Attempt status | Error / reason |", "| --- | --- | --- | --- |"]
        lines += [f"| {cell(r.name)} | {r.status} | {r.attempt_status or 'not submitted'} | {cell((r.error_type or '') + ': ' + (r.reason or ''))} |" for r in comparison.runs]
        for title, ids in (("Successful Runs", comparison.successful_runs), ("Failed Runs", comparison.failed_runs),
                           ("Blocked / Incomplete Runs", comparison.blocked_runs + comparison.incomplete_runs)):
            lines += ["", f"## {title}", "", ", ".join(names[i] for i in ids) or "None."]
        lines += ["", "## Metrics Table", "", "| Run | Accuracy | Macro-F1 | ECE |", "| --- | ---: | ---: | ---: |"]
        for run in comparison.runs:
            values = [f"{run.metrics[k]:.6f}" if run.run_id in comparison.successful_runs else "—" for k in ("accuracy", "macro_f1", "ece")]
            lines.append(f"| {cell(run.name)} | " + " | ".join(values) + " |")
        for key, title in (("accuracy", "Best Accuracy"), ("macro_f1", "Best Macro-F1"), ("ece", "Lowest ECE")):
            winner = comparison.metric_results[key]
            lines += ["", f"## {title}", "", "No successful evaluated runs." if winner.value is None else
                      f"{winner.value:.6f}: {', '.join(names[i] for i in winner.run_ids)}"]
        lines += ["", "## Failure Diagnosis", ""]
        diagnoses_found = False
        for run in comparison.runs:
            rows = [d for a in sorted(self.repo.list(Attempt, run_id=run.run_id), key=lambda a: a.attempt_no)
                    for d in self.repo.list(Diagnosis, attempt_id=a.id)]
            for diagnosis in rows:
                diagnoses_found = True
                lines += [f"### {cell(run.name)}", "",
                          f"Error type: {diagnosis.error_type}; mode: {cell(diagnosis.evidence_json.get('mode', 'recorded'))}", "",
                          cell(diagnosis.summary), "", "Evidence:", ""]
                lines += [f"- {cell(evidence)}" for evidence in diagnosis.evidence_json.get("evidence", [])] or ["- None."]
                lines += ["", "Retrieved docs: " + (", ".join(diagnosis.retrieved_doc_ids_json) or "None."), ""]
        if not diagnoses_found:
            lines.append("No persisted diagnosis (diagnosis may be disabled or no failed compute attempts exist).")
        lines += ["", "## Recovery History", ""]
        for summary in comparison.runs:
            run = self.repo.get(Run, summary.run_id)
            lines += [f"### {cell(run.name)}", "", "Original experiment parameters: `" +
                      cell(json.dumps(run.parameters_json, sort_keys=True)) + "`", ""]
            for attempt in sorted(self.repo.list(Attempt, run_id=run.id), key=lambda a: a.attempt_no):
                lines.append(f"- Attempt {attempt.attempt_no}: {attempt.status}; {attempt.error_type or 'no execution error'}")
                for diagnosis in self.repo.list(Diagnosis, attempt_id=attempt.id):
                    record = diagnosis.proposed_patch_json
                    if not record:
                        continue
                    lines.append(f"  Diagnosis: {diagnosis.error_type}. Decision: {diagnosis.decision}.")
                    changes = record.get("proposal", {}).get("patch", {}).get("changes", {})
                    for field, value in sorted(changes.items()):
                        lines.append(f"  Proposed retry change: {cell(field)} {cell(attempt.config_json.get(field))} -> {cell(value)}.")
                    if "decision" in record:
                        lines.append("  " + cell(record["decision"]["reason"]))
                    if "approval" in record:
                        lines.append("  Human " + ("approved" if record["approval"] else "rejected") + ".")
                    link = self.repo.latest_audit(run.experiment_id, "retry_attempt_created", run_id=run.id, attempt_id=attempt.id)
                    lines.append("  " + (f"Applied to retry attempt {link.output_json['retry_attempt_id']}." if link else "No retry attempt created."))
        lines += ["", "Retry modifications are separate from the original experimental design. "
                  "Human-approved scientific changes may invalidate direct sweep comparisons; metrics remain descriptive."]
        lines += ["", "## Limitations", "", "- Deterministic CPU toy dataset; no generalization or statistical significance claim.",
                  "- MockBackend simulates scheduling. Explicit toy artifact preparation performs real training.",
                  "- Recovery is optional and bounded; only validated config patches can create another Attempt. No source editing.",
                  "- SQLite and the injected dependencies target one local workflow invocation at a time."]
        if stop_reason:
            lines.append(f"- Workflow stopped safely: {cell(stop_reason)}. Outstanding jobs are not automatically cancelled.")
        return "\n".join(lines) + "\n"

    def save(self, experiment_id: str, report: str) -> Path:
        # Do not interpolate externally supplied IDs into paths without containment checks.
        root = self.policy.safe_path(Path("data/reports"))
        path = self.policy.safe_path(root / f"{experiment_id}.md")
        if path.parent != root:
            raise ValueError("Invalid experiment report path")
        root.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.read_text(encoding="utf-8") == report:
            return path
        temporary = self.policy.safe_path(path.with_suffix(".md.tmp"))
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(path)
        return path
