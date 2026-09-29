from ..db.models import Experiment, Run, Metric, AuditEvent
from ..db.repository import Repository
from ..models.comparison import ExperimentComparison, RunSummary, MetricWinner
from .evaluation_service import EvaluationMetrics

DIRECTIONS = {"accuracy": "highest", "macro_f1": "highest", "ece": "lowest"}
FAILED = {"FAILED", "TIMED_OUT", "CANCELLED"}


class ComparisonService:
    """Compare the latest DB-backed outcomes, preserving ties in stable run-name order."""

    def __init__(self, repository: Repository):
        self.repo = repository

    def compare(self, experiment_id: str) -> ExperimentComparison:
        self.repo.get(Experiment, experiment_id)
        summaries: list[RunSummary] = []
        successful: list[str] = []
        failed: list[str] = []
        blocked: list[str] = []
        incomplete: list[str] = []
        for run in sorted(self.repo.list(Run, experiment_id=experiment_id), key=lambda r: r.name):
            attempt = self.repo.latest_attempt(run.id)
            metrics = {} if attempt is None else {m.metric_name: m.metric_value for m in
                      self.repo.list(Metric, attempt_id=attempt.id, split="test")}
            complete = set(metrics) == set(DIRECTIONS)
            if complete:
                metrics = EvaluationMetrics.model_validate(metrics).model_dump()
            if run.status == "SUCCEEDED" and complete:
                successful.append(run.id)
            elif run.status in FAILED:
                failed.append(run.id)
            elif run.status == "BLOCKED":
                blocked.append(run.id)
            else:
                incomplete.append(run.id)
            events = sorted(self.repo.list(AuditEvent, run_id=run.id), key=lambda a: a.id)
            failures = [a.output_json for a in events if "reason" in a.output_json and a.outcome != "success"
                        and a.attempt_id in (None, attempt.id if attempt else None)]
            reason = failures[-1] if failures else {}
            summaries.append(RunSummary(
                run_id=run.id, name=run.name, status=run.status,
                attempt_id=attempt.id if attempt else None, attempt_status=attempt.status if attempt else None,
                config=run.base_config_json | run.parameters_json, metrics=metrics,
                error_type=reason.get("error_type") or (attempt.error_type if attempt else None),
                reason=reason.get("reason") or (attempt.error_message if attempt else None)))
        winners = {}
        for name, direction in DIRECTIONS.items():
            candidates = [r for r in summaries if r.run_id in successful]
            best = (max if direction == "highest" else min)([r.metrics[name] for r in candidates], default=None)
            winners[name] = MetricWinner(direction=direction, value=best,
                                         run_ids=[r.run_id for r in candidates if r.metrics[name] == best])
        status = "COMPLETE" if successful and len(successful) == len(summaries) else "PARTIAL" if successful else "NO_SUCCESS"
        return ExperimentComparison(experiment_id=experiment_id, status=status, runs=summaries,
                                    successful_runs=successful, failed_runs=failed, blocked_runs=blocked,
                                    incomplete_runs=incomplete, metric_results=winners)
