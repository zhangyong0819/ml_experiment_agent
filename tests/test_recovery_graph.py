import json
from dataclasses import replace
import pytest
from langgraph.types import Command
from test_graph import graph_env, start, execute
from ml_experiment_agent.agent.diagnosis import DiagnosisAgent
from ml_experiment_agent.db.models import Attempt, Run, Diagnosis, AuditEvent, Metric
from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer
from ml_experiment_agent.graph.dependencies import RunExecutionOptions
from ml_experiment_agent.graph.planning import FixturePlanner
from ml_experiment_agent.graph.workflow import build_experiment_graph
from ml_experiment_agent.llm.fix_proposer import FixProposer
from ml_experiment_agent.models.enums import Scenario as S
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.services.recovery_service import RecoveryService
from ml_experiment_agent.testing import FakeDiagnosisModel, FakeFixModel


def recovery_deps(root, deps, *, changes=None, scenarios=None):
    scenarios = scenarios or {"recovered": (S.CUDA_OOM, S.SUCCESS)}
    plans = [RunPlan(name=name, parameters={"learning_rate": .001},
             base_config=RunConfig(output_dir=root / "data/runs", batch_size=128, epochs=1)) for name in scenarios]
    return replace(deps, planner=FixturePlanner(ExperimentDefinition(goal="controlled recovery", runs=plans)),
        execution_options={name: RunExecutionOptions(scenario_sequence=sequence) for name, sequence in scenarios.items()},
        diagnosis_agent=DiagnosisAgent(FakeDiagnosisModel(), deps.experiment_service, mode="fake"),
        recovery_service=RecoveryService(deps.experiment_service, FixProposer(FakeFixModel(changes))))


def resume(root, deps, config, value):
    with sqlite_checkpointer(root / "data/checkpoints.sqlite", application_db=root / "data/application.sqlite") as saver:
        graph = build_experiment_graph(deps, saver)
        return graph.invoke(value, config, durability="sync")


def test_oom_recovery_graph_attempt_history_and_report(graph_env):
    root, deps = graph_env
    deps = recovery_deps(root, deps)
    initial, config = start(deps)
    result, history, _ = execute(root, deps, initial, config)
    assert result["workflow_status"] == "SUCCEEDED" and result["monitor_cycle"] == 6
    assert result["comparison"]["runs"][0]["error_type"] is None
    assert result["comparison"]["runs"][0]["reason"] is None
    attempts = sorted(deps.repository.list(Attempt), key=lambda a: a.attempt_no)
    assert [a.status for a in attempts] == ["FAILED", "SUCCEEDED"]
    assert [a.config_json["batch_size"] for a in attempts] == [128, 64]
    assert [a.config_json["gradient_accumulation_steps"] for a in attempts] == [1, 2]
    assert all(a.config_json["learning_rate"] == .001 for a in attempts)
    assert len(deps.repository.list(Metric)) == 3
    report = result["final_report"]
    assert "Recovery History" in report and "128 -> 64" in report
    assert "Original experiment parameters" in report and "Attempt 1: FAILED" in report
    assert "Attempt 2: SUCCEEDED" in report and "AUTO_RETRY" in report
    assert any(h.next == ("prepare_retry",) for h in history)
    # Manual rerun adopts persisted work and does not submit or evaluate again.
    calls = deps.evaluation_service.evaluation_calls
    repeated, _, _ = execute(root, deps, initial, config)
    assert repeated["final_report"] == report
    assert len(deps.backend.jobs) == 2 and len(deps.repository.list(Attempt)) == 2
    assert deps.evaluation_service.evaluation_calls == calls
    assert len(deps.repository.list(AuditEvent, action="job_retried")) == 1


def test_mixed_success_recovery_stop(graph_env):
    root, deps = graph_env
    deps = recovery_deps(root, deps, scenarios={"success": (S.SUCCESS,), "recovered": (S.CUDA_OOM, S.SUCCESS),
                                              "unrecoverable": (S.IMPORT_ERROR,)})
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert len(result["succeeded_run_ids"]) == 2 and len(result["failed_run_ids"]) == 1
    assert len(deps.repository.list(Attempt)) == 4
    assert len(deps.repository.list(Metric)) == 6
    assert "Partial result" in result["final_report"] and "STOP" in result["final_report"]


def test_graph_retry_exhaustion(graph_env):
    root, deps = graph_env
    deps = recovery_deps(root, deps, scenarios={"oom": (S.CUDA_OOM,)})
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "FAILED" and result["monitor_cycle"] == 9
    assert len(deps.repository.list(Attempt)) == 3
    assert len(deps.repository.list(AuditEvent, action="retry_exhausted")) == 1


def test_preempted_graph_retry(graph_env):
    root, deps = graph_env
    deps = recovery_deps(root, deps, scenarios={"preempted": (S.PREEMPTED, S.SUCCESS)})
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "SUCCEEDED"
    assert deps.recovery_service.proposer.model.calls == 0


@pytest.mark.parametrize("approved", [True, False])
def test_interrupt_resume_new_graph_instance(graph_env, approved):
    root, deps = graph_env
    deps = recovery_deps(root, deps, changes={"epochs": 2})
    initial, config = start(deps)
    result, _, saved = execute(root, deps, initial, config)
    assert saved.next == ("approval_gate",)
    payload = result["__interrupt__"][0].value
    assert payload["proposed_patch"] == {"epochs": 2}
    assert payload["current_config"] == {"epochs": 1}
    assert payload["error_type"] == "CUDA_OOM"
    assert payload["run_id"] == deps.repository.list(Run)[0].id
    assert "why_approval_required" in payload
    assert len(json.dumps(payload)) < 1500
    assert len(deps.repository.list(Attempt)) == len(deps.backend.jobs) == 1
    assert not deps.repository.list(AuditEvent, action="retry_attempt_created")
    assert not deps.repository.list(Metric)
    result = resume(root, deps, config, Command(resume=approved))
    assert result["workflow_status"] == ("SUCCEEDED" if approved else "FAILED")
    assert len(deps.repository.list(Attempt)) == len(deps.backend.jobs) == (2 if approved else 1)
    assert "Human " + ("approved" if approved else "rejected") in result["final_report"]
    result2 = resume(root, deps, config, None)
    assert result2 == result
    assert len(deps.backend.jobs) == (2 if approved else 1)
    actions = [a.action for a in deps.repository.list(AuditEvent)]
    assert actions.count("human_approval_requested") == actions.count("human_approved" if approved else "human_rejected") == 1


def test_approval_node_has_no_writes_before_interrupt(graph_env, monkeypatch):
    root, deps = graph_env
    deps = recovery_deps(root, deps, changes={"epochs": 2})
    initial, config = start(deps)
    result, _, _ = execute(root, deps, initial, config)
    before = len(deps.repository.list(AuditEvent))
    # Re-enter the interrupted node without a human response: reads only.
    result2 = resume(root, deps, config, None)
    assert result2["__interrupt__"][0].value == result["__interrupt__"][0].value
    assert len(deps.repository.list(AuditEvent)) == before
    assert len(deps.repository.list(Attempt)) == 1


def test_wrong_thread_cannot_resume_original(graph_env):
    root, deps = graph_env
    deps = recovery_deps(root, deps, changes={"epochs": 2})
    initial, config = start(deps)
    execute(root, deps, initial, config)
    with pytest.raises(ValueError):
        resume(root, deps, {"configurable": {"thread_id": "unrelated"}}, Command(resume=True))
    assert len(deps.repository.list(Attempt)) == 1


def test_non_boolean_approval_is_not_authorization(graph_env):
    root, deps = graph_env
    deps = recovery_deps(root, deps, changes={"epochs": 2})
    initial, config = start(deps)
    execute(root, deps, initial, config)
    with pytest.raises(ValueError, match="boolean"):
        resume(root, deps, config, Command(resume="true"))
    assert len(deps.repository.list(Attempt)) == 1


def test_checkpoint_crash_after_retry_creation(graph_env, monkeypatch):
    root, deps = graph_env
    deps = recovery_deps(root, deps)
    initial, config = start(deps)
    original = deps.recovery_service.prepare_retry
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("crash after retry commit")
    monkeypatch.setattr(deps.recovery_service, "prepare_retry", crash)
    with pytest.raises(RuntimeError, match="retry commit"):
        execute(root, deps, initial, config)
    assert len(deps.repository.list(Attempt)) == 2 and len(deps.backend.jobs) == 1
    monkeypatch.setattr(deps.recovery_service, "prepare_retry", original)
    result = resume(root, deps, config, None)
    assert result["workflow_status"] == "SUCCEEDED"
    assert len(deps.repository.list(Attempt)) == len(deps.backend.jobs) == 2


def test_uncertain_retry_submit_is_blocked_not_repeated(graph_env, monkeypatch):
    root, deps = graph_env
    deps = recovery_deps(root, deps)
    initial, config = start(deps)
    original = deps.backend.submit
    def crash(job):
        handle = original(job)
        if len(deps.backend.jobs) == 2:
            raise RuntimeError("crash after backend accepted retry")
        return handle
    monkeypatch.setattr(deps.backend, "submit", crash)
    with pytest.raises(RuntimeError, match="backend accepted"):
        execute(root, deps, initial, config)
    assert len(deps.backend.jobs) == 2
    monkeypatch.setattr(deps.backend, "submit", original)
    result = resume(root, deps, config, None)
    assert result["workflow_status"] == "BLOCKED"
    assert len(deps.repository.list(Attempt)) == len(deps.backend.jobs) == 2
    assert deps.repository.list(AuditEvent, action="submission_unknown")


def test_crash_after_human_approval_does_not_interrupt_again(graph_env, monkeypatch):
    root, deps = graph_env
    deps = recovery_deps(root, deps, changes={"epochs": 2})
    initial, config = start(deps)
    execute(root, deps, initial, config)
    original = deps.recovery_service.approve
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("crash after approval commit")
    monkeypatch.setattr(deps.recovery_service, "approve", crash)
    with pytest.raises(RuntimeError, match="approval commit"):
        resume(root, deps, config, Command(resume=True))
    monkeypatch.setattr(deps.recovery_service, "approve", original)
    result = resume(root, deps, config, None)
    assert result["workflow_status"] == "SUCCEEDED"
    assert len(deps.backend.jobs) == 2
    assert len(deps.repository.list(AuditEvent, action="human_approved")) == 1


def test_multiple_approvals_are_scoped_to_one_run(graph_env):
    root, deps = graph_env
    deps = recovery_deps(root, deps, changes={"epochs": 2}, scenarios={"a": (S.CUDA_OOM, S.SUCCESS), "b": (S.CUDA_OOM, S.SUCCESS)})
    initial, config = start(deps)
    result, _, _ = execute(root, deps, initial, config)
    first = result["__interrupt__"][0].value["run_id"]
    result = resume(root, deps, config, Command(resume=True))
    second = result["__interrupt__"][0].value["run_id"]
    assert first != second
    result = resume(root, deps, config, Command(resume=False))
    assert result["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert len(deps.repository.list(Attempt)) == 3


def test_monitor_budget_remains_global_across_recovery(graph_env):
    root, deps = graph_env
    deps = recovery_deps(root, deps)
    initial, config = start(deps)
    result, _, _ = execute(root, deps, initial | {"max_monitor_cycles": 4}, config)
    assert result["workflow_status"] == "STOPPED"
    assert result["monitor_cycle"] == 4 and len(deps.backend.jobs) == 2


def test_stale_approval_patch_not_applied(graph_env):
    root, deps = graph_env
    deps = recovery_deps(root, deps, changes={"epochs": 2})
    initial, config = start(deps)
    execute(root, deps, initial, config)
    row = deps.repository.list(Diagnosis)[0]
    changed = json.loads(json.dumps(row.proposed_patch_json))
    changed["proposal"]["patch"]["changes"] = {"learning_rate": .5}
    row.proposed_patch_json = changed
    deps.repository.commit()
    result = resume(root, deps, config, Command(resume=True))
    assert result["workflow_status"] == "FAILED"
    assert len(deps.repository.list(Attempt)) == 1


def test_crash_between_validation_and_approval_request_audit(graph_env, monkeypatch):
    root, deps = graph_env
    deps = recovery_deps(root, deps, changes={"epochs": 2})
    initial, config = start(deps)
    original = deps.recovery_service.audit
    def crash(attempt_id, action, outputs):
        original(attempt_id, action, outputs)
        if action == "fix_validated":
            raise RuntimeError("crash after validation commit")
    monkeypatch.setattr(deps.recovery_service, "audit", crash)
    with pytest.raises(RuntimeError, match="validation commit"):
        execute(root, deps, initial, config)
    assert not deps.repository.list(AuditEvent, action="human_approval_requested")
    monkeypatch.setattr(deps.recovery_service, "audit", original)
    result = resume(root, deps, config, None)
    assert result["__interrupt__"]
    assert len(deps.repository.list(AuditEvent, action="human_approval_requested")) == 1
    result = resume(root, deps, config, Command(resume=True))
    assert result["workflow_status"] == "SUCCEEDED"


def test_retry_preparation_filesystem_failure_isolated(graph_env, monkeypatch):
    root, deps = graph_env
    deps = recovery_deps(root, deps, scenarios={"ok": (S.SUCCESS,), "recover": (S.CUDA_OOM, S.SUCCESS)})
    original = deps.experiment_service.create_attempt
    def fail_save(*args, **kwargs):
        if kwargs.get("retry_of"):
            raise OSError("fixture config write failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(deps.experiment_service, "create_attempt", fail_save)
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert len(deps.backend.jobs) == 2 and len(result["blocked_run_ids"]) == 1
    assert len(deps.repository.list(Metric)) == 3
