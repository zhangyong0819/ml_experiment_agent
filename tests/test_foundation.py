import pytest
from pydantic import ValidationError
from ml_experiment_agent.models.experiment import RunConfig
from ml_experiment_agent.models.enums import Scenario, JobStatus
from ml_experiment_agent.backends.mock import MockBackend
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.models import Experiment, Run, Attempt


@pytest.mark.parametrize("patch", [{"learning_rate": -1}, {"batch_size": 0}, {"precision": "fp7"},
                                    {"extra_field": 1}, {"output_dir": "../outside"},
                                    {"learning_rate": float("nan")}, {"output_dir": ""},
                                    {"output_dir": "data/a?b"}, {"output_dir": "data\\..\\outside"}])
def test_config_invalid(patch):
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"output_dir": "data/output"} | patch)


@pytest.mark.parametrize("field,value", [("gpu_count", 2), ("cpu_count", 9),
    ("memory_gb", 33), ("timeout_minutes", 61), ("entrypoint", "evil.py")])
def test_policy_limits(environment, job, field, value):
    setattr(job, field, value)
    assert not environment[2].validate(job).allowed


@pytest.mark.parametrize("field", ["output_dir", "config_path", "working_directory"])
def test_policy_paths(environment, job, field):
    setattr(job, field, environment[0] / ".." / "outside")
    assert not environment[2].validate(job).allowed


def test_policy_counts_and_approval(environment, job):
    policy = environment[2]
    assert policy.validate(job).allowed
    assert not policy.validate(job, active_jobs=3).allowed
    assert not policy.validate(job, retries=3).allowed
    assert policy.validate(job, changed_fields={"learning_rate"}).requires_approval
    assert policy.validate(job, changed_fields={"batch_size"}).allowed
    assert not policy.validate(job, changed_fields={"seed"}).allowed


@pytest.mark.parametrize("scenario,terminal", [(Scenario.SUCCESS, JobStatus.SUCCEEDED),
    (Scenario.CUDA_OOM, JobStatus.FAILED), (Scenario.TIMEOUT, JobStatus.TIMED_OUT),
    (Scenario.PREEMPTED, JobStatus.FAILED), (Scenario.IMPORT_ERROR, JobStatus.FAILED),
    (Scenario.PATH_ERROR, JobStatus.FAILED), (Scenario.CONFIG_RUNTIME_ERROR, JobStatus.FAILED),
    (Scenario.RUNTIME_ERROR, JobStatus.FAILED)])
def test_mock_lifecycle(environment, job, scenario, terminal):
    backend = MockBackend(environment[2])
    job.scenario = scenario
    handle = backend.submit(job)
    assert [backend.status(handle.job_id) for _ in range(3)] == [JobStatus.QUEUED, JobStatus.RUNNING, terminal]
    assert backend.status(handle.job_id) == terminal
    assert backend.logs(handle.job_id).text
    assert backend.logs(handle.job_id, 0).text == ""
    if scenario == Scenario.CUDA_OOM:
        assert "torch.cuda.OutOfMemoryError" in backend.logs(handle.job_id).text
    backend.cancel(handle.job_id)
    assert backend.status(handle.job_id) == terminal


def test_cancel_and_unknown(environment, job):
    backend = MockBackend(environment[2])
    handle = backend.submit(job)
    backend.cancel(handle.job_id)
    backend.cancel(handle.job_id)
    assert backend.status(handle.job_id) == JobStatus.CANCELLED
    for action in (backend.status, backend.logs, backend.cancel):
        with pytest.raises(KeyError):
            action("missing")


def test_repository_roundtrip(tmp_path):
    factory = create_session_factory(tmp_path / "test.sqlite")
    with factory() as session:
        repo = Repository(session)
        exp = repo.add(Experiment(goal="test", plan_json={}))
        run = repo.add(Run(experiment_id=exp.id, name="r", parameters_json={}, base_config_json={}))
        first = repo.add(Attempt(run_id=run.id, attempt_no=1, config_json={}, status="FAILED", error_type="CUDA_OOM"))
        second = repo.add(Attempt(run_id=run.id, attempt_no=repo.next_attempt_no(run.id), config_json={}, status="SUCCEEDED"))
        repo.commit()
        ids = exp.id, run.id, first.id, second.id
    with factory() as session:
        repo = Repository(session)
        assert len(repo.get(Experiment, ids[0]).runs[0].attempts) == 2
        assert repo.get(Attempt, ids[2]).error_type == "CUDA_OOM"
        repo.get(Run, ids[1]).name = "updated"
        repo.commit()
        assert repo.list(Run, name="updated")
        repo.delete(repo.get(Attempt, ids[3]))
        repo.commit()
        with pytest.raises(KeyError):
            repo.get(Attempt, ids[3])
