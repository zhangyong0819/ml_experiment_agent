from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI
from .runner import ExperimentRunner
from .routes import router
from ..backends.mock import MockBackend
from ..backends.local import LocalBackend, LocalJob
from ..backends.lsf import LSFBackend
from ..backends.cluster import ClusterConfig
from ..models.project import ProjectRegistry
from ..graph.dependencies import WorkflowDependencies, ToyArtifactPreparer, RunExecutionOptions
from ..graph.planning import DeterministicPlanner
from ..services.experiment_service import ExperimentService
from ..services.policy_engine import PolicyEngine


def create_app(root: Path | None = None, *, dependencies=None, backend_type: str | None = None) -> FastAPI:
    """Application factory. No connection, model call or worker starts on import."""
    root = (root or Path(os.getenv("ML_AGENT_ROOT", "."))).resolve()
    backend_type = backend_type or os.getenv("ML_AGENT_BACKEND", "mock")
    if backend_type not in ("mock", "local", "lsf"):
        raise ValueError("Unsupported backend")
    cluster = ClusterConfig.from_environment() if backend_type == "lsf" else None
    if cluster and cluster.interactive_auth:
        raise ValueError("Interactive SSH belongs to CLI smoke, not the unattended API worker")
    if dependencies is None:
        from ..agent.diagnosis import DiagnosisAgent
        from ..llm.fix_proposer import FixProposer
        from ..services.recovery_service import RecoveryService
        from ..testing import FakeDiagnosisModel, FakeFixModel, FixtureEmbeddings, FAKE_EMBEDDING_ID
        from ..rag.documentation import DocumentationIndex
        registry = ProjectRegistry(root / "project_specs", root)
        policy = PolicyEngine.load(root / "configs/policy.yaml", root, registry)
        mock = MockBackend(policy)
        local_jobs: dict[str, LocalJob] = {}
        real_model = os.getenv("ML_AGENT_MODEL_MODE", "fake") == "deepseek"
        if real_model:
            from ..llm.deepseek import create_deepseek_model
            from ..llm.planning import DeepSeekExperimentPlanner
            from ..rag.embeddings import e5_embeddings, MODEL_NAME
            model = create_deepseek_model()
            index = DocumentationIndex.load(root / "data/faiss", e5_embeddings(root / "data/hf_cache"), MODEL_NAME)
            planner = DeepSeekExperimentPlanner(model, root / "data/runs")
        else:
            index = DocumentationIndex.build(root / "knowledge", FixtureEmbeddings(), FAKE_EMBEDDING_ID)
            planner = DeterministicPlanner(root / "data/runs", epochs=1)
        def dependencies(repo):
            if backend_type == "local":
                backend = LocalBackend(policy, repo, jobs=local_jobs)
                producer = None
            else:
                backend = LSFBackend(policy, repo, cluster) if cluster else mock
                producer = backend if cluster else None
            service = ExperimentService(repo, registry, policy, backend)
            if backend_type == "mock":
                producer = ToyArtifactPreparer(service)
            return WorkflowDependencies(service, planner, producer,
                diagnosis_agent=DiagnosisAgent(model if real_model else FakeDiagnosisModel(), service, index,
                                               mode="deepseek" if real_model else "fake"),
                recovery_service=RecoveryService(service, FixProposer(model if real_model else FakeFixModel())),
                default_execution_options=RunExecutionOptions(resources={"queue": cluster.queue} if cluster else {}))
    runner = ExperimentRunner(root, dependencies, backend_type=backend_type,
                              poll_interval=cluster.poll_interval_seconds if cluster else (1 if backend_type == "local" else 0))

    @asynccontextmanager
    async def lifespan(app):
        runner.start()
        yield
        runner.close()

    app = FastAPI(title="ML Experiment Agent", lifespan=lifespan)
    app.state.runner = runner
    app.include_router(router)
    return app
