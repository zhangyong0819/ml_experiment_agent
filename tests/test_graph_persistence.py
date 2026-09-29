from dataclasses import replace
import json
import sqlite3
import pytest
from test_graph import graph_env, execute, start, FixtureEvaluation
from ml_experiment_agent.db.models import Run, Attempt, Metric, AuditEvent
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.backends.mock import MockBackend
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.services.report_service import ReportService
from ml_experiment_agent.graph.dependencies import ToyArtifactPreparer
from ml_experiment_agent.graph.workflow import build_experiment_graph
from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer


def test_checkpoint_reload_new_graph_and_session(graph_env):
    root, deps = graph_env
    initial, config = start(deps)
    result, history, _ = execute(root, deps, initial, config)
    with create_session_factory(root / "data/application.sqlite")() as session:
        service = ExperimentService(Repository(session), deps.experiment_service.registry,
                                    deps.policy_engine, MockBackend(deps.policy_engine))
        fresh_deps = replace(deps, experiment_service=service, artifact_preparer=ToyArtifactPreparer(service))
        with sqlite_checkpointer(root / "data/checkpoints.sqlite", application_db=root / "data/application.sqlite") as saver:
            fresh = build_experiment_graph(fresh_deps, saver)
            assert fresh.get_state(config).values == result
            assert len(list(fresh.get_state_history(config))) == len(history)
            assert fresh.invoke(None, config) == result
    with sqlite3.connect(root / "data/application.sqlite") as connection:
        names = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "checkpoints" not in names and "experiments" in names
    with sqlite3.connect(root / "data/checkpoints.sqlite") as connection:
        names = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "checkpoints" in names and "experiments" not in names


def test_thread_isolation(graph_env):
    root, deps = graph_env
    first, first_config = start(deps, "first experiment")
    second, second_config = start(deps, "second experiment")
    a, _, _ = execute(root, deps, first, first_config)
    b, _, _ = execute(root, deps, second, second_config)
    assert set(a["succeeded_run_ids"]).isdisjoint(b["succeeded_run_ids"])
    assert a["report_path"] != b["report_path"]
    with sqlite_checkpointer(root / "data/checkpoints.sqlite", application_db=root / "data/application.sqlite") as saver:
        graph = build_experiment_graph(deps, saver)
        assert graph.get_state(first_config).values == a
        assert graph.get_state(second_config).values == b
        for config in (first_config, second_config):
            for snapshot in graph.get_state_history(config):
                json.dumps(snapshot.values)
                assert snapshot.values.get("experiment_id", config["configurable"]["thread_id"]) == config["configurable"]["thread_id"]


def test_checkpoint_databases_must_be_separate(graph_env):
    root, _ = graph_env
    with pytest.raises(ValueError, match="separate"):
        with sqlite_checkpointer(root / "same.sqlite", application_db=root / "same.sqlite"):
            pytest.fail("Must reject before opening")


def test_thread_id_mismatch(graph_env):
    root, deps = graph_env
    initial, _ = start(deps)
    with pytest.raises(ValueError, match="thread_id"):
        execute(root, deps, initial, {"configurable": {"thread_id": "different"}})
    assert not deps.repository.list(Run)


def test_resume_after_materialization_with_fresh_runtime(graph_env, monkeypatch):
    root, deps = graph_env
    initial, config = start(deps)
    def crash(_run_id):
        raise RuntimeError("simulated process failure before validation")
    monkeypatch.setattr(deps.experiment_service, "validate_run_config", crash)
    with pytest.raises(RuntimeError, match="simulated process"):
        execute(root, deps, initial, config)
    assert len(deps.repository.list(Run)) == 3 and not deps.repository.list(Attempt)
    with create_session_factory(root / "data/application.sqlite")() as session:
        repo = Repository(session)
        service = ExperimentService(repo, deps.experiment_service.registry, deps.policy_engine, MockBackend(deps.policy_engine))
        service.evaluation = FixtureEvaluation(repo, deps.policy_engine)
        fresh_deps = replace(deps, experiment_service=service, artifact_preparer=ToyArtifactPreparer(service))
        with sqlite_checkpointer(root / "data/checkpoints.sqlite", application_db=root / "data/application.sqlite") as saver:
            graph = build_experiment_graph(fresh_deps, saver)
            assert graph.get_state(config).next == ("validate_configs",)
            result = graph.invoke(None, config, durability="sync")
        assert result["workflow_status"] == "SUCCEEDED"
        assert len(repo.list(Run)) == len(repo.list(Attempt)) == 3


def test_resume_after_committed_submit_does_not_resubmit(graph_env, monkeypatch):
    root, deps = graph_env
    initial, config = start(deps)
    original = deps.experiment_service.submit_attempt
    def crash_after_commit(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("crash after application commit before checkpoint")
    monkeypatch.setattr(deps.experiment_service, "submit_attempt", crash_after_commit)
    with pytest.raises(RuntimeError, match="after application commit"):
        execute(root, deps, initial, config)
    assert len(deps.backend.jobs) == 1
    first_job = deps.repository.list(Attempt)[0].job_id
    monkeypatch.setattr(deps.experiment_service, "submit_attempt", original)
    with sqlite_checkpointer(root / "data/checkpoints.sqlite", application_db=root / "data/application.sqlite") as saver:
        graph = build_experiment_graph(deps, saver)
        assert graph.get_state(config).next == ("submit_ready_runs",)
        result = graph.invoke(None, config, durability="sync")
    assert result["workflow_status"] == "SUCCEEDED"
    assert len(deps.backend.jobs) == len(deps.repository.list(Attempt)) == 3
    assert first_job in [a.job_id for a in deps.repository.list(Attempt)]


def test_ambiguous_submit_blocks_instead_of_duplicate(graph_env, monkeypatch):
    root, deps = graph_env
    initial, config = start(deps)
    original = deps.backend.submit
    def crash_after_accept(job):
        original(job)
        raise RuntimeError("crash after external acceptance before application commit")
    monkeypatch.setattr(deps.backend, "submit", crash_after_accept)
    with pytest.raises(RuntimeError, match="external acceptance"):
        execute(root, deps, initial, config)
    assert len(deps.backend.jobs) == 1
    monkeypatch.setattr(deps.backend, "submit", original)
    with sqlite_checkpointer(root / "data/checkpoints.sqlite", application_db=root / "data/application.sqlite") as saver:
        result = build_experiment_graph(deps, saver).invoke(None, config, durability="sync")
    assert result["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert len(result["blocked_run_ids"]) == 1
    assert len(deps.backend.jobs) == 3  # one ambiguous job, two other runs; no fourth submit
    assert len(deps.repository.list(AuditEvent, action="submission_unknown")) == 1


def test_resume_with_lost_mock_jobs_is_blocked(graph_env, monkeypatch):
    root, deps = graph_env
    initial, config = start(deps)
    def crash(_job_id):
        raise RuntimeError("process died before monitoring")
    monkeypatch.setattr(deps.backend, "status", crash)
    with pytest.raises(RuntimeError, match="before monitoring"):
        execute(root, deps, initial, config)
    deps.experiment_service.backend = MockBackend(deps.policy_engine)
    with sqlite_checkpointer(root / "data/checkpoints.sqlite", application_db=root / "data/application.sqlite") as saver:
        result = build_experiment_graph(deps, saver).invoke(None, config, durability="sync")
    assert result["workflow_status"] == "BLOCKED"
    assert len(result["blocked_run_ids"]) == 3
    assert not deps.backend.jobs and len(deps.repository.list(Attempt)) == 3


def test_resume_after_metrics_commit_with_fresh_backend(graph_env, monkeypatch):
    root, deps = graph_env
    initial, config = start(deps)
    original = ReportService.save
    def crash(*_args, **_kwargs):
        raise RuntimeError("report storage programming failure")
    monkeypatch.setattr(ReportService, "save", crash)
    with pytest.raises(RuntimeError, match="report storage"):
        execute(root, deps, initial, config)
    assert len(deps.repository.list(Metric)) == 9
    monkeypatch.setattr(ReportService, "save", original)
    deps.experiment_service.backend = MockBackend(deps.policy_engine)
    with sqlite_checkpointer(root / "data/checkpoints.sqlite", application_db=root / "data/application.sqlite") as saver:
        graph = build_experiment_graph(deps, saver)
        assert graph.get_state(config).next == ("generate_report",)
        result = graph.invoke(None, config, durability="sync")
    assert result["workflow_status"] == "SUCCEEDED" and not deps.backend.jobs
    assert len(deps.repository.list(Metric)) == 9
    assert deps.evaluation_service.evaluation_calls == 3
