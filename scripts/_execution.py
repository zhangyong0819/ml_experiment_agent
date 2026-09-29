"""Shared CLI wiring for the existing graph; no separate execution workflow."""
from contextlib import contextmanager
from pathlib import Path
from ml_experiment_agent.agent.diagnosis import DiagnosisAgent
from ml_experiment_agent.backends.local import LocalBackend
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer
from ml_experiment_agent.graph.dependencies import WorkflowDependencies, RunExecutionOptions
from ml_experiment_agent.graph.planning import FixturePlanner
from ml_experiment_agent.graph.workflow import build_experiment_graph
from ml_experiment_agent.llm.fix_proposer import FixProposer
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.models.project import ProjectRegistry
from ml_experiment_agent.rag.documentation import DocumentationIndex
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.services.hpc_bridge import HPCBridge
from ml_experiment_agent.services.policy_engine import PolicyEngine
from ml_experiment_agent.services.recovery_service import RecoveryService
from ml_experiment_agent.testing import FakeDiagnosisModel, FakeFixModel, FixtureEmbeddings, FAKE_EMBEDDING_ID


def definition(root: Path, recovery: bool) -> ExperimentDefinition:
    return ExperimentDefinition(goal="Safe resource-fixture recovery" if recovery else "Real CPU toy execution",
        project_name="toy_recovery_fixture" if recovery else "toy_classification",
        runs=[RunPlan(name="one", parameters={"learning_rate": .001},
                      base_config=RunConfig(output_dir=root / "data/runs", batch_size=128, epochs=1))])


@contextmanager
def execution(root: Path, mode: str, *, memory_unit=None, recovery=False, queue="734090d!"):
    registry = ProjectRegistry(root / "project_specs", root)
    policy = PolicyEngine.load(root / "configs/policy.yaml", root, registry)
    database = root / f"data/{mode}_execution.sqlite"
    factory = create_session_factory(database)
    try:
        with factory() as session:
            repo = Repository(session)
            backend = LocalBackend(policy, repo) if mode == "local" else HPCBridge(policy, repo, memory_unit=memory_unit)
            service = ExperimentService(repo, registry, policy, backend)
            index = DocumentationIndex.build(root / "knowledge", FixtureEmbeddings(), FAKE_EMBEDDING_ID)
            deps = WorkflowDependencies(service, FixturePlanner(definition(root, recovery)),
                diagnosis_agent=DiagnosisAgent(FakeDiagnosisModel(), service, index, mode="fake"),
                recovery_service=RecoveryService(service, FixProposer(FakeFixModel())),
                default_execution_options=RunExecutionOptions(resources={"queue": queue} if mode == "hpc" else {}))
            with sqlite_checkpointer(root / f"data/{mode}_execution_checkpoints.sqlite", application_db=database) as saver:
                yield repo, service, build_experiment_graph(deps, saver, deferred_polling=True)
    finally:
        factory.kw["bind"].dispose()
