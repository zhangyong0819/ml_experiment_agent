from pathlib import Path
from subprocess import SubprocessError
from ...db.models import Experiment, Artifact
from ...models.enums import ArtifactType, ErrorType
from ...models.experiment import ArtifactRecord
from ..dependencies import WorkflowDependencies
from ..state import ExperimentGraphState
from .common import audit_once, fail_run, require_attempt, runs, snapshot

EXPECTED_ERRORS = (ValueError, OSError, SubprocessError)


def prepare_artifacts(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    """An explicit producer bridge for MockBackend; optional for real compute backends."""
    if deps.artifact_preparer is not None:
        for run in runs(deps, state):
            if run.status != "SUCCEEDED":
                continue
            attempt = require_attempt(deps, run)
            try:
                deps.artifact_preparer.prepare(attempt.id)
            except EXPECTED_ERRORS as exc:
                deps.repository.rollback()
                fail_run(deps, run, "artifact_preparation_failed", str(exc), ErrorType.RUNTIME_ERROR)
    return snapshot(deps, state)


def discover_artifacts(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    experiment = deps.repository.get(Experiment, state["experiment_id"])
    spec = deps.experiment_service.registry.get(experiment.plan_json["project_name"])
    for run in runs(deps, state):
        if run.status != "SUCCEEDED":
            continue
        attempt = require_attempt(deps, run)
        output = Path(attempt.config_json["output_dir"])
        try:
            logs = deps.repository.list(Artifact, attempt_id=attempt.id, artifact_type=ArtifactType.LOG.value)
            if not logs or not (output / "job.log").is_file():
                deps.experiment_service.collect_logs(attempt.id)
            # Always rediscover even with metrics: deleted checkpoints must not be silently repaired.
            records = [ArtifactRecord(attempt_id=attempt.id, artifact_type=kind, path=output / name)
                       for kind, name in ((ArtifactType.CONFIG, "config.yaml"), (ArtifactType.CHECKPOINT, spec.checkpoint_name),
                                          (ArtifactType.LOG, "job.log"))]
            deps.experiment_service.record_artifacts(attempt.id, records)
        except (KeyError, *EXPECTED_ERRORS) as exc:
            deps.repository.rollback()
            fail_run(deps, run, "artifact_missing", str(exc), ErrorType.PATH_ERROR)
            continue
        audit_once(deps, run.experiment_id, "artifact_discovered", run_id=run.id, attempt_id=attempt.id,
                   outputs={"types": [r.artifact_type.value for r in records]})
    return snapshot(deps, state)


def evaluate_runs(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    for run in runs(deps, state):
        if run.status != "SUCCEEDED":
            continue
        attempt = require_attempt(deps, run)
        try:
            metrics = deps.experiment_service.evaluate_attempt(attempt.id, prepare_artifacts=False)
        except EXPECTED_ERRORS as exc:
            deps.repository.rollback()
            fail_run(deps, run, "evaluation_failed", str(exc), ErrorType.RUNTIME_ERROR)
            continue
        values = {m.metric_name: m.metric_value for m in metrics}
        audit_once(deps, run.experiment_id, "evaluation_completed", run_id=run.id, attempt_id=attempt.id)
        audit_once(deps, run.experiment_id, "metrics_saved", run_id=run.id, attempt_id=attempt.id, outputs=values)
    return snapshot(deps, state)
