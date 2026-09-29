import json
from dataclasses import replace
from pathlib import Path
import pytest
from ml_experiment_agent.db.models import Experiment, Run, Attempt, Metric, Artifact, AuditEvent
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig, MetricRecord
from ml_experiment_agent.models.enums import Scenario, JobStatus
from ml_experiment_agent.backends.mock import MockBackend
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.services.evaluation_service import EvaluationService
from ml_experiment_agent.graph.dependencies import WorkflowDependencies, ToyArtifactPreparer, RunExecutionOptions
from ml_experiment_agent.graph.planning import DeterministicPlanner, FixturePlanner
from ml_experiment_agent.graph.workflow import build_experiment_graph
from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer
from ml_experiment_agent.graph.nodes.planning import plan_experiment
from ml_experiment_agent.graph.nodes.materialization import materialize_runs
from ml_experiment_agent.graph.nodes.validation import validate_configs, policy_check
from ml_experiment_agent.graph.nodes.execution import submit_ready_runs


class FixtureEvaluation(EvaluationService):
    """Fast orchestration fixture. Real PyTorch graph integration is tested separately."""
    def __init__(self, repository, policy):
        super().__init__(repository, policy)
        self.training_calls = 0
        self.evaluation_calls = 0
        self.fail_outputs: set[str] = set()

    def prepare_artifacts(self, spec, config_path):
        self.training_calls += 1
        output = self._check_output_paths(spec, config_path)
        (output / spec.checkpoint_name).write_text("fixture checkpoint")
        (output / spec.metadata_name).write_text("{}")

    def evaluate_existing(self, attempt_id, spec, config_path):
        self.evaluation_calls += 1
        output = self._check_output_paths(spec, config_path)
        if str(output) in self.fail_outputs:
            raise ValueError("Fixture evaluation failure")
        if not (output / spec.checkpoint_name).is_file():
            raise FileNotFoundError("Missing fixture checkpoint")
        values = {"accuracy": 0.8, "macro_f1": 0.7, "ece": 0.1}
        (output / spec.evaluation_name).write_text(json.dumps(values))
        result = [MetricRecord(attempt_id=attempt_id, metric_name=k, metric_value=v) for k, v in values.items()]
        for metric in result:
            self.repository.save_metric(**metric.model_dump())
        return result


@pytest.fixture
def graph_env(environment):
    root, registry, policy = environment
    with create_session_factory(root / "data/application.sqlite")() as session:
        repo = Repository(session)
        service = ExperimentService(repo, registry, policy, MockBackend(policy))
        service.evaluation = FixtureEvaluation(repo, policy)
        deps = WorkflowDependencies(service, DeterministicPlanner(root / "data/runs", epochs=1), ToyArtifactPreparer(service))
        yield root, deps


def start(deps, goal="Compare fixture learning rates"):
    experiment = deps.experiment_service.create_experiment(ExperimentDefinition(goal=goal))
    return {"experiment_id": experiment.id, "user_goal": goal}, {"configurable": {"thread_id": experiment.id}}


def execute(root, deps, initial=None, config=None):
    if initial is None:
        initial, config = start(deps)
    with sqlite_checkpointer(root / "data/checkpoints.sqlite", application_db=root / "data/application.sqlite") as saver:
        graph = build_experiment_graph(deps, saver)
        result = graph.invoke(initial, config, durability="sync")
        return result, list(graph.get_state_history(config)), graph.get_state(config)


def stage(deps, initial, config):
    state = initial | plan_experiment(initial, config, deps)
    for node in (materialize_runs, validate_configs, policy_check):
        state |= node(state, deps)
    return state


def test_graph_compile_and_mermaid(graph_env):
    _, deps = graph_env
    graph = build_experiment_graph(deps)
    mermaid = graph.get_graph().draw_mermaid()
    for node in ("plan_experiment", "monitor_runs", "evaluate_runs", "finalize_experiment"):
        assert node in mermaid
    assert "monitor_runs -." in mermaid


def test_all_success_and_checkpoint_state(graph_env):
    root, deps = graph_env
    result, history, saved = execute(root, deps)
    assert result["workflow_status"] == "SUCCEEDED"
    assert result["monitor_cycle"] == 3
    assert len(result["succeeded_run_ids"]) == 3
    assert not result["failed_run_ids"] and not result["active_run_ids"]
    assert saved.values == result and not saved.next
    assert len(history) > 10
    assert json.loads(json.dumps(result)) == result
    assert "repository" not in result and "backend" not in result
    assert len(deps.repository.list(Metric)) == 9
    assert len(deps.repository.list(Artifact)) == 15
    assert deps.evaluation_service.training_calls == 3
    assert deps.evaluation_service.evaluation_calls == 3
    assert Path(result["report_path"]).read_text(encoding="utf-8") == result["final_report"]
    assert deps.repository.get(Experiment, result["experiment_id"]).status == "SUCCEEDED"
    actions = {a.action for a in deps.repository.list(AuditEvent)}
    assert {"plan_created", "runs_materialized", "config_validated", "policy_approved", "job_submitted",
            "job_status_changed", "artifact_discovered", "evaluation_completed", "metrics_saved",
            "comparison_completed", "report_generated", "experiment_finalized"} <= actions
    states = [h.values for h in reversed(history) if h.values.get("monitor_cycle")]
    assert {s["monitor_cycle"] for s in states} == {1, 2, 3}


def test_mixed_partial(graph_env):
    root, deps = graph_env
    deps = replace(deps, execution_options={"run_lr_0001": RunExecutionOptions(Scenario.CUDA_OOM)})
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert len(result["succeeded_run_ids"]) == 2 and len(result["failed_run_ids"]) == 1
    assert result["comparison"]["status"] == "PARTIAL"
    assert "Partial result" in result["final_report"] and "CUDA_OOM" in result["final_report"]
    failed = deps.repository.latest_attempt(result["failed_run_ids"][0])
    assert failed.status == "FAILED" and failed.error_type == "CUDA_OOM"
    assert deps.repository.list(Artifact, attempt_id=failed.id)
    assert len(deps.repository.list(Metric)) == 6


def test_all_failed(graph_env):
    root, deps = graph_env
    names = ["run_lr_001", "run_lr_0005", "run_lr_0001"]
    deps = replace(deps, execution_options={n: RunExecutionOptions(Scenario.CUDA_OOM) for n in names})
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "FAILED"
    assert result["comparison"]["status"] == "NO_SUCCESS"
    assert result["comparison"]["metric_results"]["accuracy"]["value"] is None
    assert not deps.repository.list(Metric)
    assert "No successful evaluated runs" in result["final_report"]


def test_materialize_and_submit_idempotency(graph_env):
    _, deps = graph_env
    initial, config = start(deps)
    state = stage(deps, initial, config)
    ids = list(state["ready_run_ids"])
    materialize_runs(state, deps)
    assert len(deps.repository.list(Run)) == 3
    first = submit_ready_runs(state, deps)
    second = submit_ready_runs(state, deps)  # stale checkpoint with old ready IDs
    assert first == second
    assert set(first["active_run_ids"]) == set(ids)
    assert len(deps.backend.jobs) == len(deps.repository.list(Attempt)) == 3


def test_whole_graph_rerun_idempotency(graph_env):
    root, deps = graph_env
    initial, config = start(deps)
    first, _, _ = execute(root, deps, initial, config)
    path = Path(first["report_path"])
    modified = path.stat().st_mtime_ns
    second, _, _ = execute(root, deps, initial, config)
    assert first == second
    assert path.stat().st_mtime_ns == modified
    assert len(deps.repository.list(Run)) == len(deps.repository.list(Attempt)) == len(deps.backend.jobs) == 3
    assert len(deps.repository.list(Metric)) == 9
    assert len(deps.repository.list(Artifact)) == 15
    assert deps.evaluation_service.training_calls == deps.evaluation_service.evaluation_calls == 3
    assert len(deps.repository.list(AuditEvent, action="report_generated")) == 1


def test_invalid_config_isolated(graph_env):
    root, deps = graph_env
    plan = deps.planner.plan("unused", deps.experiment_service.registry.get("toy_classification"))
    plan.runs[0].parameters["learning_rate"] = -1
    deps = replace(deps, planner=FixturePlanner(plan))
    result, _, _ = execute(root, deps)
    assert len(result["failed_run_ids"]) == 1 and len(result["succeeded_run_ids"]) == 2
    assert len(deps.backend.jobs) == 2
    assert deps.repository.latest_attempt(result["failed_run_ids"][0]) is None
    assert deps.repository.list(AuditEvent, action="config_rejected")


def test_policy_rejection_isolated(graph_env):
    root, deps = graph_env
    deps = replace(deps, execution_options={"run_lr_001": RunExecutionOptions(resources={"gpu_count": 2})})
    result, _, _ = execute(root, deps)
    assert len(result["blocked_run_ids"]) == 1 and len(deps.backend.jobs) == 2
    assert result["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert deps.repository.latest_attempt(result["blocked_run_ids"][0]) is None
    assert deps.repository.list(AuditEvent, action="policy_rejected")


def test_all_policy_rejected(graph_env):
    root, deps = graph_env
    deps.policy_engine.config.max_gpu_count = 0
    deps = replace(deps, execution_options={n: RunExecutionOptions(resources={"gpu_count": 1})
                                           for n in ("run_lr_001", "run_lr_0005", "run_lr_0001")})
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "BLOCKED" and not deps.backend.jobs
    assert not deps.repository.list(Attempt)


def test_bounded_monitor_loop(graph_env, monkeypatch):
    root, deps = graph_env
    monkeypatch.setattr(deps.backend, "status", lambda job_id: JobStatus.RUNNING)
    initial, config = start(deps)
    result, history, saved = execute(root, deps, initial | {"max_monitor_cycles": 30}, config)
    assert result["workflow_status"] == "STOPPED" and result["monitor_cycle"] == 30
    assert result["stop_reason"] and len(result["active_run_ids"]) == 3
    assert not saved.next and len(history) > 30
    assert deps.repository.get(Experiment, initial["experiment_id"]).status == "BLOCKED"
    assert all(a.status == "RUNNING" for a in deps.repository.list(Attempt))


def test_parallel_capacity_uses_graph_waves(graph_env):
    root, deps = graph_env
    deps.policy_engine.config.max_parallel_jobs = 1
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "SUCCEEDED" and result["monitor_cycle"] == 9
    assert len(deps.backend.jobs) == 3


def test_missing_checkpoint_does_not_evaluate(graph_env):
    root, deps = graph_env
    deps = replace(deps, artifact_preparer=None)
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "FAILED"
    assert deps.evaluation_service.evaluation_calls == 0
    assert not deps.repository.list(Metric)
    assert len(deps.repository.list(AuditEvent, action="artifact_missing")) == 3
    assert all(a.status == "SUCCEEDED" for a in deps.repository.list(Attempt))


def test_deleted_checkpoint_not_recreated_on_rerun(graph_env):
    root, deps = graph_env
    initial, config = start(deps)
    result, _, _ = execute(root, deps, initial, config)
    attempt = deps.repository.latest_attempt(result["succeeded_run_ids"][0])
    (Path(attempt.config_json["output_dir"]) / "checkpoint.pt").unlink()
    rerun, _, _ = execute(root, deps, initial, config)
    assert rerun["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert deps.evaluation_service.training_calls == 3
    assert len(rerun["comparison"]["successful_runs"]) == 2


def test_evaluation_failure_isolated(graph_env, monkeypatch):
    root, deps = graph_env
    original = deps.evaluation_service.evaluate_existing
    calls = 0
    def evaluate(attempt_id, spec, path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("expected bad metrics fixture")
        return original(attempt_id, spec, path)
    monkeypatch.setattr(deps.evaluation_service, "evaluate_existing", evaluate)
    initial, config = start(deps)
    result, _, _ = execute(root, deps, initial, config)
    assert result["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert len(result["failed_run_ids"]) == 1 and len(deps.repository.list(Metric)) == 6
    assert len(deps.repository.list(AuditEvent, action="evaluation_failed")) == 1
    repeated, _, _ = execute(root, deps, initial, config)
    assert repeated["final_report"] == result["final_report"]
    assert not deps.repository.list(AuditEvent, action="job_failed")


def test_consumed_monitor_budget_never_submits_more_work(graph_env):
    root, deps = graph_env
    deps.policy_engine.config.max_parallel_jobs = 1
    initial, config = start(deps)
    first, _, _ = execute(root, deps, initial | {"max_monitor_cycles": 1}, config)
    assert first["workflow_status"] == "STOPPED" and len(deps.backend.jobs) == 1
    repeated, _, _ = execute(root, deps, initial, config)
    assert repeated["workflow_status"] == "STOPPED" and len(deps.backend.jobs) == 1


def test_cached_metrics_do_not_hide_config_tampering(graph_env):
    root, deps = graph_env
    initial, config = start(deps)
    first, _, _ = execute(root, deps, initial, config)
    attempt = deps.repository.latest_attempt(first["succeeded_run_ids"][0])
    path = Path(attempt.config_json["output_dir"]) / "config.yaml"
    path.write_text("output_dir: data/other\nunknown_field: 1")
    repeated, _, _ = execute(root, deps, initial, config)
    assert repeated["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert len(repeated["failed_run_ids"]) == 1
    assert deps.evaluation_service.evaluation_calls == 3


def test_real_cpu_end_to_end_graph(graph_env):
    root, deps = graph_env
    deps.experiment_service.evaluation = EvaluationService(deps.repository, deps.policy_engine)
    deps = replace(deps, execution_options={"run_lr_0001": RunExecutionOptions(Scenario.CUDA_OOM)})
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert len(deps.repository.list(Metric)) == 6
    import torch
    for run_id in result["succeeded_run_ids"]:
        attempt = deps.repository.latest_attempt(run_id)
        checkpoint = torch.load(Path(attempt.config_json["output_dir"]) / "checkpoint.pt", weights_only=True)
        assert checkpoint["state_dict"]
