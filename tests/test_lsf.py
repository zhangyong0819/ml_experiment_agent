import json
from types import SimpleNamespace
import pytest
from ml_experiment_agent.backends.cluster import ClusterConfig
from ml_experiment_agent.backends.lsf import LSFBackend
from ml_experiment_agent.backends.remote_operations import execute
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.db.models import Attempt
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.services.experiment_service import ExperimentService
from test_graph import graph_env, start
from ml_experiment_agent.graph.workflow import build_experiment_graph
from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer


@pytest.fixture
def lsf(environment, monkeypatch):
    root, registry, policy = environment
    factory = create_session_factory(root / "application.sqlite")
    with factory() as session:
        repo = Repository(session)
        cluster = ClusterConfig(ssh_host_alias="school", remote_project_root="/work/agent", memory_unit="MB")
        backend = LSFBackend(policy, repo, cluster)
        service = ExperimentService(repo, registry, policy, backend)
        exp = service.create_experiment(ExperimentDefinition(goal="test"))
        run = service.create_runs(exp.id, [RunPlan(name="one", base_config=RunConfig(output_dir=root / "data/run"))])[0]
        attempt = service.create_attempt(run.id)
        calls = []
        def remote(operation, **kw):
            calls.append((operation, kw))
            return {"returncode": 0, "stdout": "Job <123> is submitted to queue <734090d!>."}
        monkeypatch.setattr(backend, "_remote", remote)
        yield service, backend, attempt, calls
    factory.kw["bind"].dispose()


def submit(lsf):
    service, backend, attempt, calls = lsf
    service.submit_attempt(attempt.id, queue="734090d!")
    return backend, attempt, calls


def test_submit_command_and_durable_identity(lsf):
    backend, attempt, calls = submit(lsf)
    assert attempt.job_id == "123" and attempt.backend == "lsf"
    assert calls[0][1]["cpu"] == 1 and calls[0][1]["memory"] == 2048
    fresh = LSFBackend(backend.policy, backend.repo, backend.cluster)
    assert fresh._record("123").attempt_id == attempt.id
    with pytest.raises(ValueError):
        backend.submit(lsf[0].build_job_spec(attempt.run_id, config=RunConfig.model_validate(attempt.config_json), queue="734090d!"))


@pytest.mark.parametrize("status,expected", [("PEND", "QUEUED"), ("RUN", "RUNNING"), ("DONE", "SUCCEEDED"),
    ("EXIT", "FAILED"), ("PSUSP", "QUEUED"), ("USUSP", "RUNNING"), ("SSUSP", "RUNNING")])
def test_status_mapping(lsf, monkeypatch, status, expected):
    backend, _, _ = submit(lsf)
    monkeypatch.setattr(backend, "_remote", lambda *a, **kw: {"returncode": 0, "stdout": status})
    assert backend.status("123").value == expected


@pytest.mark.parametrize("status", ["UNKWN", "ZOMBI", "", "RUN\nDONE"])
def test_unknown_status_not_success(lsf, monkeypatch, status):
    backend, _, _ = submit(lsf)
    monkeypatch.setattr(backend, "_remote", lambda *a, **kw: {"returncode": 0, "stdout": status})
    with pytest.raises(OSError):
        backend.status("123")


@pytest.mark.parametrize("operation", ["status", "logs", "cancel"])
def test_unknown_job(lsf, operation):
    with pytest.raises(KeyError):
        getattr(lsf[1], operation)("999")
    assert not lsf[3]


@pytest.mark.parametrize("resources", [{"queue": "bad"}, {"queue": "q;id"}, {"gpu_count": 2},
    {"cpu_count": 9}, {"memory_gb": 33}, {"timeout_minutes": 61}])
def test_limits_before_ssh(lsf, resources):
    service, _, attempt, calls = lsf
    with pytest.raises(ValueError):
        service.submit_attempt(attempt.id, **({"queue": "734090d!"} | resources))
    assert not calls


def test_logs_cancel(lsf, monkeypatch):
    backend, _, _ = submit(lsf)
    calls = []
    def remote(operation, **kw):
        calls.append((operation, kw))
        return {"returncode": 0, "stdout": "RUN", "text": "training log"}
    monkeypatch.setattr(backend, "_remote", remote)
    assert backend.logs("123", 5).text == "training log"
    backend.cancel("123")
    assert calls[-1] == ("cancel", {"job_id": "123"})
    assert calls[0][1]["output"].startswith("data/run/")


def test_bad_bsub_response(lsf, monkeypatch):
    monkeypatch.setattr(lsf[1], "_remote", lambda *a, **kw: {"returncode": 0, "stdout": "submitted maybe 123"})
    with pytest.raises(OSError, match="reconciled"):
        lsf[0].submit_attempt(lsf[2].id, queue="734090d!")
    assert lsf[2].job_id is None
    with pytest.raises(OSError, match="reconciliation"):
        lsf[0].submit_attempt(lsf[2].id, queue="734090d!")


def test_fixed_remote_command_generation(tmp_path, monkeypatch):
    calls = []
    def run(args, **kw):
        calls.append((args, kw))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr("ml_experiment_agent.backends.remote_operations.subprocess.run", run)
    result = execute({"operation": "submit", "root": str(tmp_path), "working": "toy_project",
        "entrypoint": "toy_project/train.py", "output": "data/attempt", "config": {},
        "queue": "734090d!", "cpu": 1, "gpu": 0, "timeout": 2, "memory": 1024})
    args, kw = calls[0]
    assert args[:9] == ["bsub", "-q", "734090d!", "-n", "1", "-W", "2", "-R", "rusage[mem=1024]"]
    assert "shell" not in kw and result["returncode"] == 0
    assert json.loads((tmp_path / "data/attempt/config.yaml").read_text())["output_dir"] == str(tmp_path / "data/attempt")


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", "data/../../outside", "data;id"])
def test_remote_traversal(tmp_path, path):
    with pytest.raises(ValueError):
        execute({"operation": "logs", "root": str(tmp_path), "output": path, "tail": 5})


def test_deferred_poll_checkpoint_resume(graph_env):
    root, deps = graph_env
    state, config = start(deps)
    with sqlite_checkpointer(root / "poll.sqlite", application_db=root / "app.sqlite") as saver:
        graph = build_experiment_graph(deps, saver, deferred_polling=True)
        first = graph.invoke(state | {"max_monitor_cycles": 1}, config)
        assert first["active_run_ids"] and graph.get_state(config).next == ("wait_for_poll",)
        graph = build_experiment_graph(deps, saver, deferred_polling=True)
        second = graph.invoke(None, config)
        assert second["active_run_ids"] and not second.get("stop_reason")
        final = graph.invoke(None, config)
        assert final["workflow_status"] == "SUCCEEDED"
        assert len(deps.repository.list(Attempt)) == 3
