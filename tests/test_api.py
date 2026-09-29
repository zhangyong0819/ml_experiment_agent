import time
import pytest
from fastapi.testclient import TestClient
from ml_experiment_agent.api.app import create_app
from ml_experiment_agent.agent.diagnosis import DiagnosisAgent
from ml_experiment_agent.backends.mock import MockBackend
from ml_experiment_agent.graph.dependencies import WorkflowDependencies, ToyArtifactPreparer, RunExecutionOptions
from ml_experiment_agent.graph.planning import FixturePlanner
from ml_experiment_agent.llm.fix_proposer import FixProposer
from ml_experiment_agent.models.enums import Scenario
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.services.recovery_service import RecoveryService
from ml_experiment_agent.testing import FakeDiagnosisModel, FakeFixModel
from test_graph import FixtureEvaluation


def app_for(environment, *, human=False):
    root, registry, policy = environment
    backend = MockBackend(policy)
    def dependencies(repo):
        service = ExperimentService(repo, registry, policy, backend)
        service.evaluation = FixtureEvaluation(repo, policy)
        definition = ExperimentDefinition(goal="fixture", runs=[RunPlan(name="one", base_config=RunConfig(output_dir=root / "data/runs"))])
        return WorkflowDependencies(service, FixturePlanner(definition), ToyArtifactPreparer(service),
            {"one": RunExecutionOptions(scenario_sequence=(Scenario.CUDA_OOM, Scenario.SUCCESS))} if human else {},
            DiagnosisAgent(FakeDiagnosisModel(), service, None, mode="fake"),
            RecoveryService(service, FixProposer(FakeFixModel({"epochs": 2}))))
    return create_app(root, dependencies=dependencies)


def wait(client, identifier, wanted):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = client.get(f"/experiments/{identifier}")
        assert result.status_code == 200, result.text
        if result.json()["status"] in wanted:
            return result.json()
        time.sleep(.02)
    pytest.fail(f"No expected status: {result.json()}")


def test_api_lifecycle(environment):
    with TestClient(app_for(environment)) as client:
        assert client.get("/health").json() == {"status": "ok", "backend": "mock"}
        response = client.post("/experiments", json={"goal": "one quick run"})
        assert response.status_code == 202
        identifier = response.json()["experiment_id"]
        result = wait(client, identifier, {"SUCCEEDED"})
        run = client.get("/runs/" + result["runs"][0]["id"]).json()
        assert len(run["attempts"]) == 1
        assert len(client.get(f"/experiments/{identifier}/metrics").json()) == 3
        report = client.get(f"/experiments/{identifier}/report").json()
        # DB success may precede graph report by a few milliseconds.
        for _ in range(100):
            if report["ready"]:
                break
            time.sleep(.01)
            report = client.get(f"/experiments/{identifier}/report").json()
        assert report["ready"] and "one quick run" in report["report"]
        assert client.post(f"/experiments/{identifier}/resume", json={"approved": True}).status_code == 409


@pytest.mark.parametrize("approved", [True, False])
def test_api_hitl(environment, approved):
    with TestClient(app_for(environment, human=True)) as client:
        identifier = client.post("/experiments", json={"goal": "test approval"}).json()["experiment_id"]
        result = wait(client, identifier, {"AWAITING_APPROVAL"})
        assert result["approval"]["proposed_patch"] == {"epochs": 2}
        assert len(result["runs"][0]["attempts"]) == 1
        assert client.post(f"/experiments/{identifier}/resume", json={"approved": approved}).status_code == 202
        result = wait(client, identifier, {"SUCCEEDED"} if approved else {"FAILED"})
        assert len(result["runs"][0]["attempts"]) == (2 if approved else 1)


@pytest.mark.parametrize("body", [{"goal": ""}, {"goal": "x", "shell": "id"},
    {"goal": "x", "entrypoint": "evil.py"}, {"goal": "x", "output_dir": "/etc"},
    {"goal": "x", "project_name": "unknown"}])
def test_api_invalid_request(environment, body):
    with TestClient(app_for(environment)) as client:
        assert client.post("/experiments", json=body).status_code == 422


@pytest.mark.parametrize("path", ["/experiments/missing", "/runs/missing", "/experiments/missing/metrics", "/experiments/missing/report"])
def test_api_not_found(environment, path):
    with TestClient(app_for(environment)) as client:
        assert client.get(path).status_code == 404


def test_resume_strict_boolean(environment):
    with TestClient(app_for(environment)) as client:
        assert client.post("/experiments/missing/resume", json={"approved": "true"}).status_code == 422
