"""Graph diagnosis demo: MODE=fake without key, MODE=deepseek with explicit environment key."""
import argparse
import os
from pathlib import Path
from ml_experiment_agent.agent.diagnosis import DiagnosisAgent
from ml_experiment_agent.backends.mock import MockBackend
from ml_experiment_agent.db.models import Diagnosis, AuditEvent, Run
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer
from ml_experiment_agent.graph.dependencies import WorkflowDependencies, ToyArtifactPreparer, RunExecutionOptions
from ml_experiment_agent.graph.planning import DeterministicPlanner
from ml_experiment_agent.graph.workflow import build_experiment_graph
from ml_experiment_agent.llm.deepseek import create_deepseek_model
from ml_experiment_agent.llm.planning import DeepSeekExperimentPlanner
from ml_experiment_agent.models.enums import Scenario
from ml_experiment_agent.models.experiment import ExperimentDefinition
from ml_experiment_agent.models.project import ProjectRegistry
from ml_experiment_agent.rag.documentation import DocumentationIndex
from ml_experiment_agent.rag.embeddings import e5_embeddings, MODEL_NAME
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.services.policy_engine import PolicyEngine
from ml_experiment_agent.settings import Settings
from ml_experiment_agent.testing import FakeDiagnosisModel, FixtureEmbeddings, FAKE_EMBEDDING_ID


def demo(*, force_fake: bool = False) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings.from_root(root)
    real = bool(os.getenv("DEEPSEEK_API_KEY", "").strip()) and not force_fake
    mode = "deepseek" if real else "fake"
    print(f"MODE={mode}", flush=True)
    model = create_deepseek_model() if real else FakeDiagnosisModel()
    if real:
        embedding = e5_embeddings(root / "data/hf_cache")
        index = DocumentationIndex.load(root / "data/faiss", embedding, MODEL_NAME)
        planner = DeepSeekExperimentPlanner(model, root / "data/runs")
    else:
        index = DocumentationIndex.build(root / "knowledge", FixtureEmbeddings(), FAKE_EMBEDDING_ID)
        index.save(root / "data/faiss_fake")
        planner = DeterministicPlanner(root / "data/runs")
        print("EMBEDDING=fixture-token-hash-v1 (not real E5)", flush=True)
    registry = ProjectRegistry(root / "project_specs", root)
    policy = PolicyEngine.load(root / "configs/policy.yaml", root, registry)
    with create_session_factory(settings.database_path)() as session:
        repo = Repository(session)
        service = ExperimentService(repo, registry, policy, MockBackend(policy))
        goal = ("Compare exactly three learning rates: run_lr_001=0.001, run_lr_0005=0.0005, "
                "run_lr_0001=0.0001; compare accuracy, macro_f1, ece. Use exactly those run names.")
        experiment = service.create_experiment(ExperimentDefinition(goal=goal))
        deps = WorkflowDependencies(service, planner, ToyArtifactPreparer(service),
            {"run_lr_0001": RunExecutionOptions(Scenario.CUDA_OOM)}, DiagnosisAgent(model, service, index, mode=mode))
        checkpoint_path = root / "data/langgraph_checkpoints.sqlite"
        config = {"configurable": {"thread_id": experiment.id}}
        with sqlite_checkpointer(checkpoint_path, application_db=settings.database_path) as saver:
            result = build_experiment_graph(deps, saver).invoke({"experiment_id": experiment.id, "user_goal": goal}, config, durability="sync")
        print(f"Experiment={experiment.id} status={result['workflow_status']} monitor_cycles={result['monitor_cycle']}")
        for run in repo.list(Run, experiment_id=experiment.id):
            attempt = repo.latest_attempt(run.id)
            print(f"{run.name}: {run.status}; attempt={attempt.id if attempt else None}")
            for diagnosis in repo.list(Diagnosis, attempt_id=attempt.id) if attempt else []:
                print(f"  diagnosis={diagnosis.error_type}: {diagnosis.summary}")
                print(f"  retrieved_docs={diagnosis.retrieved_doc_ids_json}")
        calls = repo.list(AuditEvent, experiment_id=experiment.id, action="tool_called")
        print("Tools:", [c.input_json.get("tool", c.output_json.get("tool")) for c in calls])
        print(f"Report={result['report_path']}\nCheckpoint={checkpoint_path}")
        if not result["diagnosed_run_ids"]:
            raise RuntimeError("Demo produced no diagnosis; inspect planner run names and failure scenario")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="Force offline fake mode even if an API key exists")
    demo(force_fake=parser.parse_args().fake)
