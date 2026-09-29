"""Real lightweight child processes through the production API dependency wiring."""
import json
from pathlib import Path
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from ml_experiment_agent.api.app import create_app
from ml_experiment_agent.backends.local import LocalBackend
from ml_experiment_agent.db.models import Attempt, AuditEvent
from ml_experiment_agent.graph.planning import DeterministicPlanner
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunConfig, RunPlan
from ml_experiment_agent.testing import FakeFixModel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api_root(environment, monkeypatch):
    root, _, _ = environment
    shutil.copytree(ROOT / "knowledge", root / "knowledge")
    monkeypatch.setenv("ML_AGENT_MODEL_MODE", "fake")
    monkeypatch.delenv("ML_AGENT_BACKEND", raising=False)
    return root


def test_api_default_backend_remains_mock(api_root):
    with TestClient(create_app(api_root)) as client:
        assert client.get("/health").json() == {"status": "ok", "backend": "mock"}
        assert client.post("/experiments", json={"goal": "x", "backend": "local"}).status_code == 422


def test_api_invalid_backend_rejected(api_root, monkeypatch):
    monkeypatch.setenv("ML_AGENT_BACKEND", "unknown")
    with pytest.raises(ValueError, match="Unsupported backend"):
        create_app(api_root)


@pytest.mark.parametrize("mode", ["success", "recovery", "approve", "reject"])
def test_api_local_real_process_workflow(api_root, monkeypatch, mode):
    root = api_root
    monkeypatch.setenv("ML_AGENT_BACKEND", "local")
    # Only plan/model fixtures are replaced; API -> runner -> graph -> LocalBackend
    # and the actual training/evaluation subprocess paths remain real.
    def plan(self, goal, spec):
        return ExperimentDefinition(goal=goal, project_name=spec.project_name, runs=[
            RunPlan(name="one", parameters={"learning_rate": .001},
                    base_config=RunConfig(output_dir=root / "data/runs", epochs=1, batch_size=128))])
    monkeypatch.setattr(DeterministicPlanner, "plan", plan)
    human = mode in {"approve", "reject"}
    if human:
        import ml_experiment_agent.llm.fix_proposer as proposer_module
        original = proposer_module.FixProposer
        monkeypatch.setattr(proposer_module, "FixProposer", lambda model: original(
            FakeFixModel({"batch_size": 64, "gradient_accumulation_steps": 2, "epochs": 2})))
    # Wait for the test to release the task: POST must return before training can finish.
    (root / "toy_project/train.py").write_text(
        "import json, os, pathlib, sys, time, yaml\n"
        "root = pathlib.Path(__file__).resolve().parents[1]\n"
        "config = yaml.safe_load(pathlib.Path(sys.argv[2]).read_text())\n"
        "output = pathlib.Path(config['output_dir'])\n"
        "print('REAL_LOCAL_PID=' + str(os.getpid()), flush=True)\n"
        "deadline = time.monotonic() + 15\n"
        "while not (root / 'release').exists():\n"
        "    if time.monotonic() > deadline: raise RuntimeError('fixture release timeout')\n"
        "    time.sleep(.02)\n"
        f"if {mode != 'success'!r} and config['batch_size'] > 64:\n"
        "    print('torch.cuda.OutOfMemoryError: CUDA out of memory (API fixture)', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "(output / 'checkpoint.pt').write_bytes(b'lightweight API process fixture')\n"
        "(output / 'metadata.json').write_text(json.dumps({'pid': os.getpid()}))\n",
        encoding="utf-8")
    (root / "toy_project/evaluate.py").write_text(
        "import json, pathlib, sys, yaml\n"
        "config = yaml.safe_load(pathlib.Path(sys.argv[2]).read_text())\n"
        "output = pathlib.Path(config['output_dir'])\n"
        "assert (output / 'checkpoint.pt').is_file()\n"
        "(output / 'metrics.json').write_text(json.dumps({'accuracy': .8, 'macro_f1': .75, 'ece': .1}))\n",
        encoding="utf-8")
    app = create_app(root)
    runner = app.state.runner
    assert runner.poll_interval == 1
    runner.poll_interval = .05  # Same due-time scheduler, shorter test interval.
    with runner.context() as (_, deps, _):
        backend = deps.backend
        assert isinstance(backend, LocalBackend)
        assert deps.artifact_preparer is None
    with runner.context() as (_, other, _):
        assert other.backend is not backend
        assert other.backend.repo is not backend.repo
        assert other.backend.jobs is backend.jobs

    def wait(client, identifier, wanted, *, report=False):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            response = client.get(f"/experiments/{identifier}")
            assert response.status_code == 200
            result = response.json()
            if result["status"] in wanted:
                if not report or client.get(f"/experiments/{identifier}/report").json()["ready"]:
                    return result
            time.sleep(.02)
        pytest.fail(f"API workflow did not reach {wanted}: {result}")

    try:
        with TestClient(app) as client:
            assert client.get("/health").json() == {"status": "ok", "backend": "local"}
            assert not backend.jobs  # Startup/health do not launch a process.
            response = client.post("/experiments", json={"goal": "local API execution", "project_name": "toy_classification"})
            assert response.status_code == 202
            identifier = response.json()["experiment_id"]
            wait(client, identifier, {"RUNNING"})
            assert not client.get(f"/experiments/{identifier}/report").json()["ready"]
            (root / "release").touch()
            if human:
                awaiting = wait(client, identifier, {"AWAITING_APPROVAL"})
                assert awaiting["approval"]["proposed_patch"]["epochs"] == 2
                assert len(awaiting["runs"][0]["attempts"]) == 1
                assert len(backend.jobs) == 1
                assert client.post(f"/experiments/{identifier}/resume", json={"approved": mode == "approve"}).status_code == 202
            result = wait(client, identifier, {"FAILED"} if mode == "reject" else {"SUCCEEDED"}, report=True)
            attempts = result["runs"][0]["attempts"]
            count = 1 if mode in {"success", "reject"} else 2
            assert len(attempts) == len(backend.jobs) == count
            assert all(a["backend"] == "local" and a["job_id"].startswith("local-") for a in attempts)
            if mode != "success":
                assert attempts[0]["status"] == "FAILED"
                assert attempts[0]["diagnoses"][0]["error_type"] == "CUDA_OOM"
            metrics = client.get(f"/experiments/{identifier}/metrics").json()
            assert len(metrics) == (0 if mode == "reject" else 3)
            report = client.get(f"/experiments/{identifier}/report").json()
            assert "local API execution" in report["report"] and "Recovery History" in report["report"]
            with runner.context() as (repo, _, _):
                records = sorted(repo.list(Attempt), key=lambda a: a.attempt_no)
                if count == 2:
                    assert [a.config_json["batch_size"] for a in records] == [128, 64]
                    assert repo.list(AuditEvent, action="job_retried")
                for attempt in records:
                    output = Path(attempt.config_json["output_dir"])
                    assert "REAL_LOCAL_PID=" in (output / "stdout.log").read_text()
                    if attempt.status == "SUCCEEDED":
                        assert json.loads((output / "metadata.json").read_text())["pid"] > 0
    finally:
        (root / "release").touch()
        for job in backend.jobs.values():
            if job.process.poll() is None:
                backend._terminate(job)
