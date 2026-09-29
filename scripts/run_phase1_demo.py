"""Run three deterministic mock jobs and evaluate the two successful CPU models."""
from pathlib import Path
from ml_experiment_agent.settings import Settings
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.models.project import ProjectRegistry
from ml_experiment_agent.models.enums import Scenario
from ml_experiment_agent.backends.mock import MockBackend
from ml_experiment_agent.services.policy_engine import PolicyEngine
from ml_experiment_agent.services.experiment_service import ExperimentService


def main() -> None:
    settings = Settings.from_root(Path(__file__).resolve().parents[1])
    registry = ProjectRegistry(settings.root / "project_specs", settings.root)
    policy = PolicyEngine.load(settings.root / "configs/policy.yaml", settings.root, registry)
    factory = create_session_factory(settings.database_path)
    with factory() as session:
        service = ExperimentService(Repository(session), registry, policy, MockBackend(policy))
        plans = [RunPlan(name=name, parameters={"learning_rate": rate},
                         base_config=RunConfig(output_dir=settings.root / "data/runs"))
                 for name, rate in [("run_lr_001", 0.01), ("run_lr_0005", 0.005), ("run_lr_0001", 0.001)]]
        experiment = service.create_experiment(ExperimentDefinition(goal="Compare three learning rates", runs=plans))
        print(f"Experiment: {experiment.id}")
        for run, scenario in zip(service.create_runs(experiment.id), [Scenario.SUCCESS, Scenario.SUCCESS, Scenario.CUDA_OOM]):
            attempt = service.create_attempt(run.id)
            service.submit_attempt(attempt.id, scenario)
            for _ in range(3):
                service.refresh_attempt_status(attempt.id)
            logs = service.collect_logs(attempt.id)
            print(f"{run.name}: run={run.id} attempt={attempt.id} status={attempt.status}")
            if attempt.status == "SUCCEEDED":
                print("  metrics:", {m.metric_name: round(m.metric_value, 6) for m in service.evaluate_attempt(attempt.id)})
            else:
                print(f"  error={attempt.error_type}: {logs.text.splitlines()[-1]}")
        print(f"Experiment status: {experiment.status}")
    print(f"Database: {settings.database_path}")


if __name__ == "__main__":
    main()
