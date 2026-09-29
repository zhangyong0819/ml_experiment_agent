import json
import subprocess
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError
from ml_experiment_agent.db.models import Experiment, Attempt, Metric, Artifact, AuditEvent, Diagnosis
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.backends.mock import MockBackend
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig, ArtifactRecord
from ml_experiment_agent.models.enums import Scenario, ArtifactType
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.services.evaluation_service import EvaluationMetrics


@pytest.fixture
def service(environment):
    root, registry, policy = environment
    factory = create_session_factory(root / "data/test.sqlite")
    with factory() as session:
        yield ExperimentService(Repository(session), registry, policy, MockBackend(policy))


def prepare(service, root, count=1):
    plans = [RunPlan(name=f"run_{i}", base_config=RunConfig(output_dir=root / "data/runs", epochs=1),
                     parameters={"learning_rate": 0.01}) for i in range(count)]
    exp = service.create_experiment(ExperimentDefinition(goal="integration", runs=plans))
    return exp, service.create_runs(exp.id)


def finish(service, attempt, scenario=Scenario.SUCCESS):
    service.submit_attempt(attempt.id, scenario)
    assert [service.refresh_attempt_status(attempt.id).status for _ in range(3)] == [
        "QUEUED", "RUNNING", "SUCCEEDED" if scenario == Scenario.SUCCESS else "FAILED"]


def test_end_to_end(service, environment):
    exp, runs = prepare(service, environment[0])
    attempt = service.create_attempt(runs[0].id)
    finish(service, attempt)
    assert attempt.submitted_at and attempt.started_at and attempt.ended_at
    assert "Training complete" in service.collect_logs(attempt.id).text
    metrics = service.evaluate_attempt(attempt.id)
    assert {m.metric_name for m in metrics} == {"accuracy", "macro_f1", "ece"}
    assert all(0 <= m.metric_value <= 1 for m in metrics)
    assert service.evaluate_attempt(attempt.id) == metrics
    assert len(service.repo.list(Metric, attempt_id=attempt.id)) == 3
    artifacts = service.repo.list(Artifact, attempt_id=attempt.id)
    assert len(artifacts) == 5
    assert all(Path(a.path).is_file() for a in artifacts)
    assert exp.status == "SUCCEEDED"
    actions = {a.action for a in service.repo.list(AuditEvent, experiment_id=exp.id)}
    assert {"create_experiment", "create_run", "create_attempt", "submit_attempt", "refresh_status",
            "collect_logs", "evaluate_attempt"} <= actions
    assert not service.repo.list(Diagnosis)
    factory = create_session_factory(environment[0] / "data/test.sqlite")
    with factory() as session:
        persisted = Repository(session)
        assert persisted.get(Experiment, exp.id).status == "SUCCEEDED"
        assert len(persisted.list(Metric, attempt_id=attempt.id)) == 3
        assert len(persisted.list(Artifact, attempt_id=attempt.id)) == 5


def test_failure_and_manual_attempt_history(service, environment):
    exp, runs = prepare(service, environment[0])
    first = service.create_attempt(runs[0].id)
    finish(service, first, Scenario.CUDA_OOM)
    assert first.error_type == "CUDA_OOM" and exp.status == "FAILED"
    with pytest.raises(ValueError):
        service.evaluate_attempt(first.id)
    second = service.create_attempt(runs[0].id)
    assert second.attempt_no == 2 and second.config_json["output_dir"] != first.config_json["output_dir"]
    finish(service, second)
    assert first.status == "FAILED" and exp.status == "SUCCEEDED"
    assert len(service.repo.list(Attempt, run_id=runs[0].id)) == 2


def test_submit_rejection_and_audit(service, environment, monkeypatch):
    exp, runs = prepare(service, environment[0])
    attempt = service.create_attempt(runs[0].id)
    with pytest.raises(ValueError):
        service.submit_attempt(attempt.id, gpu_count=2)
    assert attempt.status == "CREATED" and not service.backend.jobs
    assert service.repo.list(AuditEvent, outcome="rejected")
    with pytest.raises(ValueError):
        service.create_attempt(runs[0].id)
    def forbidden_submit(_job):
        pytest.fail("Validation scenarios must never reach backend.submit")
    monkeypatch.setattr(service.backend, "submit", forbidden_submit)
    with pytest.raises(ValueError):
        service.submit_attempt(attempt.id, Scenario.INVALID_CONFIG)
    with pytest.raises(ValueError):
        service.submit_attempt(attempt.id, Scenario.POLICY_VIOLATION)


def test_config_tamper_rejected(service, environment):
    _, runs = prepare(service, environment[0])
    attempt = service.create_attempt(runs[0].id)
    path = Path(attempt.config_json["output_dir"]) / "config.yaml"
    path.write_text("batch_size: 0\noutput_dir: data/test")
    with pytest.raises(ValueError):
        service.submit_attempt(attempt.id)
    assert not service.backend.jobs


def test_cancel_service(service, environment):
    _, runs = prepare(service, environment[0])
    attempt = service.create_attempt(runs[0].id)
    service.submit_attempt(attempt.id)
    assert service.cancel_attempt(attempt.id).status == "CANCELLED"
    with pytest.raises(ValueError):
        service.submit_attempt(attempt.id)


def test_parallel_limit(service, environment):
    _, runs = prepare(service, environment[0], 4)
    attempts = [service.create_attempt(r.id) for r in runs]
    for a in attempts[:3]:
        service.submit_attempt(a.id)
    with pytest.raises(ValueError, match="max_parallel_jobs"):
        service.submit_attempt(attempts[3].id)
    service.cancel_attempt(attempts[0].id)
    service.submit_attempt(attempts[3].id)


def test_retry_limit(service, environment):
    _, runs = prepare(service, environment[0])
    for _ in range(3):
        attempt = service.create_attempt(runs[0].id)
        finish(service, attempt, Scenario.CUDA_OOM)
    with pytest.raises(ValueError, match="max_retries"):
        service.create_attempt(runs[0].id)


@pytest.mark.parametrize("values", [{"accuracy": 2, "macro_f1": 0, "ece": 0},
    {"accuracy": 0.5, "macro_f1": 0.5}, {"accuracy": 0.5, "macro_f1": 0.5, "ece": float("nan")},
    {"accuracy": 0.5, "macro_f1": 0.5, "ece": 0, "extra": 1}])
def test_metrics_schema(values):
    with pytest.raises(ValidationError):
        EvaluationMetrics.model_validate(values)


def test_artifact_reject_external(service, environment):
    _, runs = prepare(service, environment[0])
    attempt = service.create_attempt(runs[0].id)
    other = environment[0] / "other.txt"
    other.write_text("outside attempt")
    with pytest.raises(ValueError):
        service.record_artifacts(attempt.id, [ArtifactRecord(attempt_id=attempt.id,
            artifact_type=ArtifactType.METADATA, path=other)])


def test_training_deterministic_and_ece(environment, job):
    import importlib.util
    import torch
    root = environment[0]
    sys.path.insert(0, str(root / "toy_project"))
    try:
        module_spec = importlib.util.spec_from_file_location("toy_evaluate", root / "toy_project/evaluate.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        assert module.expected_calibration_error(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1])) == 0
        assert module.expected_calibration_error(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([1, 0])) == 1
    finally:
        sys.path.pop(0)
    states, metrics = [], []
    for _ in range(2):
        for entry in ("train.py", "evaluate.py"):
            subprocess.run([sys.executable, str(root / "toy_project" / entry), "--config", str(job.config_path)],
                           check=True, capture_output=True, timeout=60)
        states.append(torch.load(job.output_dir / "checkpoint.pt", weights_only=True)["state_dict"])
        metrics.append(json.loads((job.output_dir / "metrics.json").read_text()))
    assert metrics[0] == metrics[1]
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
