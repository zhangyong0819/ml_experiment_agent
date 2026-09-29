import os
import subprocess
from pathlib import Path
import pytest
from ml_experiment_agent.backends.mock import MockBackend
from ml_experiment_agent.services.config_service import ConfigService
from ml_experiment_agent.models.experiment import RunConfig


def test_resolved_link_escape(environment, job, tmp_path):
    root, _, policy = environment
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = root / "escape"
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                       check=True, capture_output=True)
    else:
        link.symlink_to(outside, target_is_directory=True)
    assert link.resolve() == outside.resolve()
    job.output_dir = link / "output"
    assert not policy.validate(job).allowed
    with pytest.raises(ValueError, match="outside"):
        policy.safe_path(link / "checkpoint.pt")


def test_direct_backend_config_path_mismatch(environment, job):
    ConfigService().save(RunConfig(output_dir=environment[0] / "different"), job.config_path)
    backend = MockBackend(environment[2])
    with pytest.raises(ValueError, match="does not match"):
        backend.submit(job)
    assert not backend.jobs


def test_unregistered_project(environment, job):
    job.project_name = "arbitrary"
    assert not environment[2].validate(job).allowed
    with pytest.raises(ValueError, match="Unregistered"):
        MockBackend(environment[2]).submit(job)


def test_backend_revalidates_config(environment, job):
    job.config_path.write_text("output_dir: data/test\nunknown: 3")
    with pytest.raises(ValueError):
        MockBackend(environment[2]).submit(job)


def test_worker_budget(environment, job):
    ConfigService().save(RunConfig(output_dir=job.output_dir, num_workers=8), job.config_path)
    with pytest.raises(ValueError, match="cpu_count"):
        MockBackend(environment[2]).submit(job)
