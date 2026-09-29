from dataclasses import replace
import pytest
from test_graph import graph_env, execute, start, stage
from ml_experiment_agent.db.models import Run, Metric, AuditEvent
from ml_experiment_agent.models.policy import PolicyDecision
from ml_experiment_agent.models.experiment import ExperimentDefinition
from ml_experiment_agent.services.comparison_service import ComparisonService
from ml_experiment_agent.services.report_service import ReportService
from ml_experiment_agent.graph.nodes.execution import submit_ready_runs


def test_python_comparison_directions_ties_and_report(graph_env):
    root, deps = graph_env
    result, _, _ = execute(root, deps)
    repo = deps.repository
    service = ComparisonService(repo)
    tied = service.compare(result["experiment_id"])
    assert len(tied.metric_results["accuracy"].run_ids) == 3
    ordered = sorted(repo.list(Run), key=lambda r: r.name)
    values = [(0.9, 0.6, 0.3), (0.7, 0.95, 0.2), (0.8, 0.7, 0.05)]
    for run, row in zip(ordered, values):
        attempt = repo.latest_attempt(run.id)
        for metric, value in zip(("accuracy", "macro_f1", "ece"), row):
            repo.save_metric(attempt_id=attempt.id, metric_name=metric, metric_value=value, split="test")
    repo.commit()
    comparison = service.compare(result["experiment_id"])
    assert comparison.metric_results["accuracy"].run_ids == [ordered[0].id]
    assert comparison.metric_results["macro_f1"].run_ids == [ordered[1].id]
    assert comparison.metric_results["ece"].run_ids == [ordered[2].id]
    report = ReportService(repo, deps.policy_engine)
    first = report.render(comparison)
    assert first == report.render(service.compare(result["experiment_id"]))
    for heading in ("Experiment Goal", "Run Summary", "Config Differences", "Execution Status", "Successful Runs",
                    "Failed Runs", "Metrics Table", "Best Accuracy", "Best Macro-F1", "Lowest ECE", "Limitations"):
        assert heading in first


def test_empty_comparison_is_explicit(graph_env):
    _, deps = graph_env
    initial, _ = start(deps)
    comparison = ComparisonService(deps.repository).compare(initial["experiment_id"])
    assert comparison.status == "NO_SUCCESS" and not comparison.runs
    assert comparison.metric_results["ece"].value is None


def test_approval_is_never_implicitly_allowed(graph_env, monkeypatch):
    root, deps = graph_env
    # Even a contradictory external decision cannot bypass requires_approval.
    monkeypatch.setattr(deps.policy_engine, "validate", lambda *a, **kw:
                        PolicyDecision(allowed=True, requires_approval=True, reason="Approval required"))
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "BLOCKED"
    assert len(result["blocked_run_ids"]) == 3 and not deps.backend.jobs


def test_injected_planner_is_only_called_once(graph_env):
    root, deps = graph_env
    delegate = deps.planner
    class CountingPlanner:
        calls = 0
        def plan(self, goal, project_spec):
            self.calls += 1
            plan = delegate.plan(goal, project_spec)
            plan.runs = plan.runs[:1]
            return plan
    planner = CountingPlanner()
    deps = replace(deps, planner=planner)
    initial, config = start(deps)
    execute(root, deps, initial, config)
    result, _, _ = execute(root, deps, initial, config)
    assert planner.calls == 1 and len(result["succeeded_run_ids"]) == 1


@pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5])
def test_invalid_monitor_bounds(graph_env, limit):
    root, deps = graph_env
    initial, config = start(deps)
    with pytest.raises(ValueError, match="Monitor bounds"):
        execute(root, deps, initial | {"max_monitor_cycles": limit}, config)
    assert not deps.backend.jobs


def test_materialization_rejects_identity_conflict(graph_env):
    _, deps = graph_env
    initial, config = start(deps)
    state = stage(deps, initial, config)
    run = deps.repository.list(Run)[0]
    run.parameters_json = {"learning_rate": 0.99}
    deps.repository.commit()
    from ml_experiment_agent.graph.nodes.materialization import materialize_runs
    with pytest.raises(RuntimeError, match="immutable plan"):
        materialize_runs(state, deps)


def test_reuse_created_attempt_without_duplicate(graph_env):
    _, deps = graph_env
    initial, config = start(deps)
    state = stage(deps, initial, config)
    run_id = state["ready_run_ids"][0]
    attempt = deps.experiment_service.create_attempt(run_id)
    submit_ready_runs(state, deps)
    assert deps.repository.latest_attempt(run_id).id == attempt.id
    assert attempt.job_id
