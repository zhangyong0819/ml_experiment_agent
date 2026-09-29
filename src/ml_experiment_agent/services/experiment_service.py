import logging
from pathlib import Path
from typing import Any
from ..backends.base import ComputeBackend
from ..db.models import Experiment, Run, Attempt, Metric, now, identifier
from ..db.repository import Repository
from ..models.experiment import ExperimentDefinition, RunPlan, RunConfig, ArtifactRecord, MetricRecord
from ..models.enums import Scenario, ArtifactType, ErrorType
from ..models.job import JobSpec, JobLog
from ..models.project import ProjectRegistry
from .config_service import ConfigService
from .policy_engine import PolicyEngine
from .artifact_service import ArtifactService
from .evaluation_service import EvaluationService

logger = logging.getLogger(__name__)
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "TIMED_OUT"}


def classify_error(text: str) -> ErrorType:
    """Classify known log signatures only; no diagnosis or remediation."""
    for signature, kind in (("OutOfMemoryError", ErrorType.CUDA_OOM), ("config error", ErrorType.CONFIG_ERROR),
                            ("ImportError", ErrorType.IMPORT_ERROR), ("FileNotFoundError", ErrorType.PATH_ERROR),
                            ("TIME LIMIT", ErrorType.TIMEOUT), ("PREEMPTED", ErrorType.PREEMPTED),
                            ("RuntimeError", ErrorType.RUNTIME_ERROR)):
        if signature in text:
            return kind
    return ErrorType.UNKNOWN


class ExperimentService:
    """Single-process application boundary. Attempts are append-only execution history."""

    def __init__(self, repository: Repository, registry: ProjectRegistry, policy: PolicyEngine,
                 backend: ComputeBackend):
        self.repo, self.registry, self.policy, self.backend = repository, registry, policy, backend
        self.configs = ConfigService()
        self.artifacts = ArtifactService(repository, policy)
        self.evaluation = EvaluationService(repository, policy)

    def create_experiment(self, definition: ExperimentDefinition) -> Experiment:
        self.registry.get(definition.project_name)
        result = self.repo.add(Experiment(goal=definition.goal, plan_json=definition.model_dump(mode="json")))
        self.repo.audit(result.id, "create_experiment", inputs=result.plan_json)
        self.repo.commit()
        return result

    def create_runs(self, experiment_id: str, plans: list[RunPlan] | None = None) -> list[Run]:
        exp = self.repo.get(Experiment, experiment_id)
        definition = ExperimentDefinition.model_validate(exp.plan_json)
        plans = definition.runs if plans is None else plans
        spec = self.registry.get(definition.project_name)
        names = [p.name for p in plans]
        existing = {r.name for r in self.repo.list(Run, experiment_id=experiment_id)}
        if len(set(names)) != len(names) or existing.intersection(names):
            raise ValueError("Run names must be unique within experiment")
        for plan in plans:
            config = self.configs.generate(plan.base_config, plan.parameters, spec)
            self.policy.safe_path(config.output_dir)
        result = []
        for plan in plans:
            run = self.repo.add(Run(experiment_id=exp.id, name=plan.name, parameters_json=plan.parameters,
                                    base_config_json=plan.base_config.model_dump(mode="json")))
            self.repo.audit(exp.id, "create_run", run_id=run.id, inputs=plan.model_dump(mode="json"))
            result.append(run)
        self.repo.commit()
        return result

    def _context(self, attempt_id: str):
        attempt = self.repo.get(Attempt, attempt_id)
        run = self.repo.get(Run, attempt.run_id)
        exp = self.repo.get(Experiment, run.experiment_id)
        spec = self.registry.get(exp.plan_json["project_name"])
        return attempt, run, exp, spec

    def materialize_run(self, experiment_id: str, plan: RunPlan) -> Run:
        """Idempotently register a candidate; config/policy checks happen before execution."""
        exp = self.repo.get(Experiment, experiment_id)
        self.registry.get(exp.plan_json["project_name"])
        existing = self.repo.list(Run, experiment_id=experiment_id, name=plan.name)
        if existing:
            run = existing[0]
            if run.parameters_json != plan.parameters or run.base_config_json != plan.base_config.model_dump(mode="json"):
                raise RuntimeError(f"Materialized run differs from immutable plan: {plan.name}")
            return run
        run = self.repo.add(Run(experiment_id=experiment_id, name=plan.name,
                               parameters_json=plan.parameters, base_config_json=plan.base_config.model_dump(mode="json")))
        self.repo.audit(exp.id, "create_run", run_id=run.id, inputs=plan.model_dump(mode="json"))
        self.repo.commit()
        return run

    def validate_run_config(self, run_id: str) -> RunConfig:
        run = self.repo.get(Run, run_id)
        exp = self.repo.get(Experiment, run.experiment_id)
        spec = self.registry.get(exp.plan_json["project_name"])
        return self.configs.generate(RunConfig.model_validate(run.base_config_json), run.parameters_json, spec)

    def build_job_spec(self, run_id: str, *, config: RunConfig | None = None,
                       scenario: Scenario = Scenario.SUCCESS, **resources: Any) -> JobSpec:
        """Build a structured preview without creating an Attempt or writing files."""
        run = self.repo.get(Run, run_id)
        exp = self.repo.get(Experiment, run.experiment_id)
        spec = self.registry.get(exp.plan_json["project_name"])
        config = (config or self.validate_run_config(run_id)).model_copy(deep=True)
        config.output_dir = self.policy.safe_path(config.output_dir)
        return self._job(spec, config, scenario, **resources)

    def _job(self, spec, config: RunConfig, scenario=Scenario.SUCCESS, **resources: Any) -> JobSpec:
        resources.setdefault("cpu_count", config.num_workers + 1)
        if resources["cpu_count"] < config.num_workers + 1:
            raise ValueError("cpu_count must cover the training process and data workers")
        return JobSpec(project_name=spec.project_name, entrypoint=spec.train_entrypoint,
                       config_path=config.output_dir / "config.yaml", working_directory=spec.working_directory,
                       output_dir=config.output_dir, scenario=scenario, **resources)

    def create_attempt(self, run_id: str, *, config: RunConfig | None = None,
                       retry_of: str | None = None) -> Attempt:
        """Append an attempt; a retry link and its record commit atomically for replay safety."""
        run = self.repo.get(Run, run_id)
        if retry_of is not None:
            source = self.repo.get(Attempt, retry_of)
            if source.run_id != run_id or source.status not in {"FAILED", "TIMED_OUT"}:
                raise ValueError("Retry source must be a failed attempt of this run")
            link = self.repo.latest_audit(run.experiment_id, "retry_attempt_created", run_id=run_id, attempt_id=retry_of)
            if link:
                return self.repo.get(Attempt, link.output_json["retry_attempt_id"])
            if self.repo.latest_attempt(run_id).id != retry_of:
                raise ValueError("Retry source is not the latest attempt")
        exp = self.repo.get(Experiment, run.experiment_id)
        spec = self.registry.get(exp.plan_json["project_name"])
        previous = self.repo.list(Attempt, run_id=run_id)
        if any(a.status not in TERMINAL for a in previous):
            raise ValueError("Previous attempt is still active")
        number = self.repo.next_attempt_no(run_id)
        config = (RunConfig.model_validate(config.model_dump()) if config is not None else
                  self.configs.generate(RunConfig.model_validate(run.base_config_json), run.parameters_json, spec))
        attempt_id = identifier()
        base_output = self.validate_run_config(run_id).output_dir
        config.output_dir = self.policy.safe_path(base_output / attempt_id)
        decision = self.policy.validate(self._job(spec, config), retries=number - 1)
        if not decision.allowed:
            self.repo.audit(exp.id, "create_attempt", run_id=run.id, outcome="rejected", outputs=decision.model_dump())
            self.repo.commit()
            raise ValueError(decision.reason)
        self.configs.save(config, config.output_dir / "config.yaml")
        attempt = self.repo.add(Attempt(id=attempt_id, run_id=run.id, attempt_no=number,
                                        config_json=config.model_dump(mode="json")))
        run.status = "CREATED"
        exp.status = "RUNNING"
        self.repo.audit(exp.id, "create_attempt", run_id=run.id, attempt_id=attempt.id)
        if retry_of is not None:
            self.repo.audit(exp.id, "retry_attempt_created", run_id=run.id, attempt_id=retry_of,
                            outputs={"retry_attempt_id": attempt.id})
        self.repo.commit()
        return attempt

    def submit_attempt(self, attempt_id: str, scenario: Scenario = Scenario.SUCCESS, **resources: Any) -> Attempt:
        attempt, run, exp, spec = self._context(attempt_id)
        if attempt.status != "CREATED":
            raise ValueError("Only CREATED attempts can be submitted")
        try:
            if scenario in (Scenario.INVALID_CONFIG, Scenario.POLICY_VIOLATION):
                raise ValueError(f"Reserved validation scenario rejected before backend submission: {scenario}")
            config = RunConfig.model_validate(attempt.config_json)
            job = self._job(spec, config, scenario, **resources)
            decision = self.policy.validate(job, active_jobs=self.repo.active_count(), retries=attempt.attempt_no - 1)
            if not decision.allowed:
                raise ValueError(decision.reason)
            disk_config = self.configs.load(self.policy.safe_path(job.config_path))
            if disk_config != config:
                raise ValueError("Persisted and on-disk configs differ")
            handle = self.backend.submit(job)
        except (ValueError, OSError) as exc:
            self.repo.audit(exp.id, "submit_attempt", run_id=run.id, attempt_id=attempt.id,
                            outcome="rejected", outputs={"error": str(exc)})
            self.repo.commit()
            raise
        attempt.job_id, attempt.backend, attempt.status = handle.job_id, handle.backend, handle.status.value
        attempt.submitted_at = now()
        self.repo.sync_status(attempt)
        self.repo.audit(exp.id, "submit_attempt", run_id=run.id, attempt_id=attempt.id, inputs=job.model_dump(mode="json"))
        self.repo.commit()
        logger.info("Submitted attempt %s as %s", attempt.id, handle.job_id)
        return attempt

    def refresh_attempt_status(self, attempt_id: str) -> Attempt:
        attempt, run, exp, _ = self._context(attempt_id)
        if attempt.status in TERMINAL:
            return attempt
        if not attempt.job_id:
            raise ValueError("Attempt has not been submitted")
        status = self.backend.status(attempt.job_id).value
        if status != attempt.status:
            attempt.status = status
            if status == "RUNNING" and attempt.started_at is None:
                attempt.started_at = now()
            if status in TERMINAL:
                attempt.ended_at = now()
                if status in ("FAILED", "TIMED_OUT"):
                    logs = self.backend.logs(attempt.job_id).text
                    attempt.error_type = classify_error(logs).value
                    attempt.error_message = logs.splitlines()[-1] if logs else "No backend error detail"
            self.repo.sync_status(attempt)
            self.repo.audit(exp.id, "refresh_status", run_id=run.id, attempt_id=attempt.id, outputs={"status": status})
            self.repo.commit()
        return attempt

    def collect_logs(self, attempt_id: str, tail: int = 200) -> JobLog:
        attempt, run, exp, _ = self._context(attempt_id)
        if not attempt.job_id:
            raise ValueError("Attempt has not been submitted")
        logs = self.backend.logs(attempt.job_id, tail)
        output = self.policy.safe_path(Path(attempt.config_json["output_dir"]))
        path = self.policy.safe_path(output / "job.log")
        path.write_text(logs.text, encoding="utf-8")
        self.artifacts.record(ArtifactRecord(attempt_id=attempt.id, artifact_type=ArtifactType.LOG, path=path), output)
        self.repo.audit(exp.id, "collect_logs", run_id=run.id, attempt_id=attempt.id)
        self.repo.commit()
        return logs

    def record_artifacts(self, attempt_id: str, records: list[ArtifactRecord]) -> None:
        attempt, run, exp, _ = self._context(attempt_id)
        for record in records:
            if record.attempt_id != attempt_id:
                raise ValueError("Artifact belongs to another attempt")
        try:
            for record in records:
                self.artifacts.record(record, Path(attempt.config_json["output_dir"]))
            self.repo.audit(exp.id, "record_artifacts", run_id=run.id, attempt_id=attempt.id)
            self.repo.commit()
        except Exception:
            self.repo.session.rollback()
            raise

    def prepare_attempt_artifacts(self, attempt_id: str) -> None:
        """Explicit mock-only training bridge, guarded by a durable completion marker."""
        attempt, run, exp, spec = self._context(attempt_id)
        if attempt.status != "SUCCEEDED":
            raise ValueError("Only successful attempts can prepare artifacts")
        if self.repo.latest_audit(exp.id, "artifacts_prepared", run_id=run.id, attempt_id=attempt.id):
            return
        output = self.policy.safe_path(Path(attempt.config_json["output_dir"]))
        config_path = self.policy.safe_path(output / "config.yaml")
        if self.configs.load(config_path).model_dump(mode="json") != attempt.config_json:
            raise ValueError("Persisted and on-disk configs differ")
        # Existing complete training output can be adopted after a crash before DB commit.
        if not all((output / name).is_file() for name in (spec.checkpoint_name, spec.metadata_name)):
            self.evaluation.prepare_artifacts(spec, config_path)
        for kind, name in ((ArtifactType.CONFIG, "config.yaml"), (ArtifactType.CHECKPOINT, spec.checkpoint_name),
                           (ArtifactType.METADATA, spec.metadata_name)):
            self.artifacts.record(ArtifactRecord(attempt_id=attempt.id, artifact_type=kind, path=output / name), output)
        self.repo.audit(exp.id, "artifacts_prepared", run_id=run.id, attempt_id=attempt.id)
        self.repo.commit()

    def evaluate_attempt(self, attempt_id: str, *, prepare_artifacts: bool = True) -> list[MetricRecord]:
        attempt, run, exp, spec = self._context(attempt_id)
        if attempt.status != "SUCCEEDED":
            raise ValueError("Only successful attempts can be evaluated")
        if not prepare_artifacts:
            output = self.policy.safe_path(Path(attempt.config_json["output_dir"]))
            checkpoint = self.policy.safe_path(output / spec.checkpoint_name)
            if not checkpoint.is_relative_to(output) or not checkpoint.is_file():
                raise FileNotFoundError(f"Missing or invalid checkpoint: {checkpoint}")
            config_path = self.policy.safe_path(output / "config.yaml")
            if self.configs.load(config_path).model_dump(mode="json") != attempt.config_json:
                raise ValueError("Persisted and on-disk configs differ")
        existing = self.repo.list(Metric, attempt_id=attempt.id)
        if existing:
            by_name = {m.metric_name: m for m in existing}
            if set(by_name) == set(spec.metric_names):
                return [MetricRecord(attempt_id=m.attempt_id, metric_name=m.metric_name,
                                 metric_value=m.metric_value, split=m.split)
                    for name in spec.metric_names for m in [by_name[name]]]
        output = self.policy.safe_path(Path(attempt.config_json["output_dir"]))
        try:
            config_path = self.policy.safe_path(output / "config.yaml")
            if self.configs.load(config_path).model_dump(mode="json") != attempt.config_json:
                raise ValueError("Persisted and on-disk configs differ")
            evaluator = self.evaluation.evaluate if prepare_artifacts else self.evaluation.evaluate_existing
            result = evaluator(attempt.id, spec, config_path)
            for kind, name in ((ArtifactType.CONFIG, "config.yaml"), (ArtifactType.CHECKPOINT, spec.checkpoint_name),
                               (ArtifactType.METADATA, spec.metadata_name), (ArtifactType.EVALUATION, spec.evaluation_name)):
                self.artifacts.record(ArtifactRecord(attempt_id=attempt.id, artifact_type=kind, path=output / name), output)
            self.repo.audit(exp.id, "evaluate_attempt", run_id=run.id, attempt_id=attempt.id,
                            outputs={m.metric_name: m.metric_value for m in result})
            self.repo.commit()
            return result
        except Exception as exc:
            self.repo.session.rollback()
            self.repo.audit(exp.id, "evaluate_attempt", run_id=run.id, attempt_id=attempt.id,
                            outcome="failed", outputs={"error": str(exc)})
            self.repo.commit()
            raise

    def cancel_attempt(self, attempt_id: str) -> Attempt:
        attempt, run, exp, _ = self._context(attempt_id)
        if not attempt.job_id:
            raise ValueError("Attempt has not been submitted")
        self.backend.cancel(attempt.job_id)
        self.repo.audit(exp.id, "cancel_attempt", run_id=run.id, attempt_id=attempt.id)
        self.repo.commit()
        return self.refresh_attempt_status(attempt_id)
