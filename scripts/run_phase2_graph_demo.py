"""All Run lifecycle operations below are driven by the compiled LangGraph."""
import json
from pathlib import Path
from ml_experiment_agent.settings import Settings
from ml_experiment_agent.db.models import Run, Attempt, Metric
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.models.experiment import ExperimentDefinition
from ml_experiment_agent.models.project import ProjectRegistry
from ml_experiment_agent.models.enums import Scenario
from ml_experiment_agent.backends.mock import MockBackend
from ml_experiment_agent.services.policy_engine import PolicyEngine
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.graph.dependencies import WorkflowDependencies, ToyArtifactPreparer, RunExecutionOptions
from ml_experiment_agent.graph.planning import DeterministicPlanner
from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer
from ml_experiment_agent.graph.workflow import build_experiment_graph


def demo() -> None:
    settings = Settings.from_root(Path(__file__).resolve().parents[1])
    registry = ProjectRegistry(settings.root / "project_specs", settings.root)
    policy = PolicyEngine.load(settings.root / "configs/policy.yaml", settings.root, registry)
    checkpoint_path = settings.root / "data/langgraph_checkpoints.sqlite"
    with create_session_factory(settings.database_path)() as session:
        repo = Repository(session)
        service = ExperimentService(repo, registry, policy, MockBackend(policy))
        goal = "Compare learning_rate=0.001, 0.0005, 0.0001 using Accuracy, Macro-F1 and ECE."
        experiment = service.create_experiment(ExperimentDefinition(goal=goal))
        deps = WorkflowDependencies(service, DeterministicPlanner(settings.root / "data/runs"),
                                    ToyArtifactPreparer(service),
                                    {"run_lr_0001": RunExecutionOptions(Scenario.CUDA_OOM)})
        config = {"configurable": {"thread_id": experiment.id}}
        with sqlite_checkpointer(checkpoint_path, application_db=settings.database_path) as saver:
            graph = build_experiment_graph(deps, saver)
            result = graph.invoke({"experiment_id": experiment.id, "user_goal": goal}, config, durability="sync")
            print(f"Experiment: {experiment.id}\nThread: {experiment.id}\nGraph status: {result['workflow_status']}")
            for run in sorted(repo.list(Run, experiment_id=experiment.id), key=lambda r: r.name):
                print(f"{run.name}: run={run.id} status={run.status}")
                for attempt in repo.list(Attempt, run_id=run.id):
                    print(f"  attempt={attempt.id} status={attempt.status} job={attempt.job_id} error={attempt.error_type}")
                    print("  metrics:", {m.metric_name: round(m.metric_value, 6) for m in repo.list(Metric, attempt_id=attempt.id)})
            print("Comparison:", json.dumps(result["comparison"]["metric_results"], indent=2))
            print(f"Report: {result['report_path']}\nReport result: {result['comparison']['status']}")
            print(f"Monitor cycles: {result['monitor_cycle']}\nCheckpoints: {len(list(graph.get_state_history(config)))}")
            print(f"Checkpoint DB: {checkpoint_path}\nApplication DB: {settings.database_path}")


if __name__ == "__main__":
    demo()
