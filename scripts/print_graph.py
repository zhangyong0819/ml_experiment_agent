"""Print public LangGraph Mermaid text without rendering or creating an experiment."""
from pathlib import Path
from sqlalchemy.orm import Session
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.models.project import ProjectRegistry
from ml_experiment_agent.backends.mock import MockBackend
from ml_experiment_agent.services.policy_engine import PolicyEngine
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.graph.planning import DeterministicPlanner
from ml_experiment_agent.graph.dependencies import WorkflowDependencies
from ml_experiment_agent.graph.workflow import build_experiment_graph


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = ProjectRegistry(root / "project_specs", root)
    policy = PolicyEngine.load(root / "configs/policy.yaml", root, registry)
    # Building topology requires no database connection and executes no nodes.
    with Session() as session:
        service = ExperimentService(Repository(session), registry, policy, MockBackend(policy))
        graph = build_experiment_graph(WorkflowDependencies(service, DeterministicPlanner(root / "data/runs")))
        print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
