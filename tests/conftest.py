from pathlib import Path
import shutil
import pytest
from ml_experiment_agent.models.project import ProjectRegistry
from ml_experiment_agent.services.policy_engine import PolicyEngine
from ml_experiment_agent.models.experiment import RunConfig
from ml_experiment_agent.models.job import JobSpec
from ml_experiment_agent.services.config_service import ConfigService

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def environment(tmp_path):
    for name in ("toy_project", "project_specs", "configs"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    registry = ProjectRegistry(tmp_path / "project_specs", tmp_path)
    policy = PolicyEngine.load(tmp_path / "configs/policy.yaml", tmp_path, registry)
    return tmp_path, registry, policy


@pytest.fixture
def job(environment):
    root, _, _ = environment
    config = RunConfig(output_dir=root / "data/output", epochs=1)
    path = root / "data/config.yaml"
    ConfigService().save(config, path)
    return JobSpec(project_name="toy_classification", entrypoint="train.py", config_path=path,
                   working_directory=root / "toy_project", output_dir=config.output_dir)
