"""Offline controlled recovery, driven entirely by the main LangGraph workflow."""
import json
from pathlib import Path
from langgraph.types import Command
from ml_experiment_agent.agent.diagnosis import DiagnosisAgent
from ml_experiment_agent.backends.mock import MockBackend
from ml_experiment_agent.db.models import Attempt, Run, Diagnosis, Metric, AuditEvent
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer
from ml_experiment_agent.graph.dependencies import WorkflowDependencies, ToyArtifactPreparer, RunExecutionOptions
from ml_experiment_agent.graph.planning import FixturePlanner
from ml_experiment_agent.graph.workflow import build_experiment_graph
from ml_experiment_agent.llm.fix_proposer import FixProposer
from ml_experiment_agent.models.enums import Scenario
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.models.project import ProjectRegistry
from ml_experiment_agent.rag.documentation import DocumentationIndex
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.services.policy_engine import PolicyEngine
from ml_experiment_agent.services.recovery_service import RecoveryService
from ml_experiment_agent.settings import Settings
from ml_experiment_agent.testing import FakeDiagnosisModel, FakeFixModel, FixtureEmbeddings, FAKE_EMBEDDING_ID


def demo(*, human: bool = False, reject: bool = False) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings.from_root(root)
    print("MODE=fake (diagnosis/proposal/embedding); real CPU toy training and evaluation", flush=True)
    registry = ProjectRegistry(root / "project_specs", root)
    policy = PolicyEngine.load(root / "configs/policy.yaml", root, registry)
    index = DocumentationIndex.build(root / "knowledge", FixtureEmbeddings(), FAKE_EMBEDDING_ID)
    goal = "Human-reviewed epoch change after OOM" if human else "Controlled OOM recovery preserving effective batch size"
    definition = ExperimentDefinition(goal=goal, runs=[RunPlan(name="run_lr_001", parameters={"learning_rate": .001},
        base_config=RunConfig(output_dir=root / "data/runs", batch_size=128, gradient_accumulation_steps=1, epochs=3))])
    with create_session_factory(settings.database_path)() as session:
        repo = Repository(session)
        service = ExperimentService(repo, registry, policy, MockBackend(policy))
        experiment = service.create_experiment(ExperimentDefinition(goal=goal))
        fix_model = FakeFixModel({"epochs": 2} if human else None)
        deps = WorkflowDependencies(service, FixturePlanner(definition), ToyArtifactPreparer(service),
            {"run_lr_001": RunExecutionOptions(scenario_sequence=(Scenario.CUDA_OOM, Scenario.SUCCESS))},
            DiagnosisAgent(FakeDiagnosisModel(), service, index, mode="fake"),
            RecoveryService(service, FixProposer(fix_model)))
        config = {"configurable": {"thread_id": experiment.id}}
        checkpoint = root / "data/langgraph_checkpoints.sqlite"
        with sqlite_checkpointer(checkpoint, application_db=settings.database_path) as saver:
            graph = build_experiment_graph(deps, saver)
            result = graph.invoke({"experiment_id": experiment.id, "user_goal": goal}, config, durability="sync")
            if human:
                if not result.get("__interrupt__"):
                    raise RuntimeError("Expected human approval interrupt")
                print("INTERRUPT=" + json.dumps(result["__interrupt__"][0].value), flush=True)
                runs = repo.list(Run, experiment_id=experiment.id)
                if any(len(repo.list(Attempt, run_id=r.id)) != 1 for r in runs):
                    raise RuntimeError("Retry created before approval")
                print(f"RESUME_APPROVED={not reject}", flush=True)
                # A fresh graph instance demonstrates checkpoint-based resume on the same thread.
                graph = build_experiment_graph(deps, saver)
                result = graph.invoke(Command(resume=not reject), config, durability="sync")
            print(f"Checkpoint count={len(list(graph.get_state_history(config)))}")
        print(f"Experiment={experiment.id} Thread={experiment.id} Final status={result['workflow_status']}")
        for run in repo.list(Run, experiment_id=experiment.id):
            print(f"Run={run.id} name={run.name} Original parameters={run.parameters_json}")
            for attempt in sorted(repo.list(Attempt, run_id=run.id), key=lambda a: a.attempt_no):
                print(f"Attempt {attempt.attempt_no}: {attempt.id} status={attempt.status} config={json.dumps(attempt.config_json, sort_keys=True)}")
                for diagnosis in repo.list(Diagnosis, attempt_id=attempt.id):
                    print(f"Diagnosis={diagnosis.error_type}; {diagnosis.summary}")
                    print("Recovery=" + json.dumps(diagnosis.proposed_patch_json, sort_keys=True))
                print("Metrics=" + json.dumps({m.metric_name: m.metric_value for m in repo.list(Metric, attempt_id=attempt.id)}))
        print("Audit actions=" + json.dumps([a.action for a in repo.list(AuditEvent, experiment_id=experiment.id)]))
        print(f"Report={result['report_path']}\nCheckpoint={checkpoint}\nMonitor cycles={result['monitor_cycle']}")
        expected = "FAILED" if human and reject else "SUCCEEDED"
        if result["workflow_status"] != expected:
            raise RuntimeError(f"Expected {expected}, got {result['workflow_status']}")


if __name__ == "__main__":
    demo()
