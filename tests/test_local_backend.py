import time
from pathlib import Path
import pytest
from ml_experiment_agent.backends.local import LocalBackend
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.services.experiment_service import ExperimentService


@pytest.fixture
def local(environment):
    root, registry, policy = environment
    factory = create_session_factory(root / 'local.sqlite')
    with factory() as session:
        repo = Repository(session)
        backend = LocalBackend(policy, repo)
        service = ExperimentService(repo, registry, policy, backend)
        exp = service.create_experiment(ExperimentDefinition(goal='process fixture'))
        run = service.create_runs(exp.id, [RunPlan(name='one', base_config=RunConfig(output_dir=root / 'data/run'))])[0]
        attempt = service.create_attempt(run.id)
        try:
            yield root, service, backend, attempt
        finally:
            for job in backend.jobs.values():
                if job.process.poll() is None:
                    backend._terminate(job)
    factory.kw['bind'].dispose()


def finish(backend, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = backend.status(job_id)
        if status.value != 'RUNNING':
            return status.value
        time.sleep(.01)
    pytest.fail('Local fixture did not terminate')


@pytest.mark.parametrize('code,expected', [(0, 'SUCCEEDED'), (3, 'FAILED')])
def test_real_process_returncode_logs_artifacts(local, code, expected):
    root, service, backend, attempt = local
    (root / 'toy_project/train.py').write_text(
        "import sys,yaml,pathlib\nc=yaml.safe_load(pathlib.Path(sys.argv[2]).read_text())\n"
        "(pathlib.Path(c['output_dir'])/'marker.txt').write_text('real process')\n"
        f"print('stdout marker');print('stderr marker',file=sys.stderr);sys.exit({code})\n")
    service.submit_attempt(attempt.id)
    assert attempt.status == 'RUNNING'
    assert finish(backend, attempt.job_id) == expected
    assert backend.status(attempt.job_id).value == expected
    log = backend.logs(attempt.job_id).text
    assert 'stdout marker' in log and 'stderr marker' in log
    assert (Path(attempt.config_json['output_dir']) / 'marker.txt').read_text() == 'real process'


def test_cancel_only_owned_process(local):
    root, service, backend, attempt = local
    (root / 'toy_project/train.py').write_text('import time\ntime.sleep(30)\n')
    service.submit_attempt(attempt.id)
    backend.cancel(attempt.job_id)
    assert backend.status(attempt.job_id).value == 'CANCELLED'
    backend.cancel(attempt.job_id)
    assert backend.jobs[attempt.job_id].process.poll() is not None


@pytest.mark.parametrize('operation', ['status', 'logs', 'cancel'])
def test_unknown_process_rejected(local, operation):
    with pytest.raises(KeyError):
        getattr(local[2], operation)('12345')


@pytest.mark.parametrize('changes', [
    {'entrypoint': 'evil.py'}, {'entrypoint': 'train.py;whoami'},
    {'working_directory': Path('../outside')}, {'config_path': Path('../config.yaml')},
    {'cpu_count': 100}, {'gpu_count': 1}])
def test_reject_unsafe_job(local, changes):
    _, service, backend, attempt = local
    job = service.build_job_spec(attempt.run_id, config=RunConfig.model_validate(attempt.config_json))
    with pytest.raises((ValueError, OSError)):
        backend.submit(job.model_copy(update=changes))
    assert not backend.jobs


def test_subprocess_uses_argv_no_shell(local, monkeypatch):
    root, service, backend, attempt = local
    import subprocess
    real = subprocess.Popen
    calls = []
    def popen(args, **kw):
        calls.append((args, kw))
        return real(args, **kw)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    (root / 'toy_project/train.py').write_text('print("ok")\n')
    service.submit_attempt(attempt.id)
    assert finish(backend, attempt.job_id) == 'SUCCEEDED'
    assert isinstance(calls[0][0], list) and calls[0][1]['shell'] is False


def test_real_timeout_and_restart_limit(local):
    root, service, backend, attempt = local
    (root / 'toy_project/train.py').write_text('import time\ntime.sleep(30)\n')
    service.submit_attempt(attempt.id, timeout_minutes=.001)
    assert finish(backend, attempt.job_id) == 'TIMED_OUT'
    fresh = LocalBackend(backend.policy, backend.repo)
    with pytest.raises(KeyError, match='reattached'):
        fresh.status(attempt.job_id)


@pytest.mark.parametrize('recovery', [False, True])
def test_real_local_graph_success_and_recovery(environment, recovery):
    from ml_experiment_agent.agent.diagnosis import DiagnosisAgent
    from ml_experiment_agent.db.models import Run, Attempt, Metric
    from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer
    from ml_experiment_agent.graph.dependencies import WorkflowDependencies
    from ml_experiment_agent.graph.planning import FixturePlanner
    from ml_experiment_agent.graph.workflow import build_experiment_graph
    from ml_experiment_agent.llm.fix_proposer import FixProposer
    from ml_experiment_agent.services.recovery_service import RecoveryService
    from ml_experiment_agent.testing import FakeDiagnosisModel, FakeFixModel
    root, registry, policy = environment
    factory = create_session_factory(root / 'real.sqlite')
    with factory() as session:
        repo = Repository(session)
        backend = LocalBackend(policy, repo)
        service = ExperimentService(repo, registry, policy, backend)
        plan = ExperimentDefinition(goal='real CPU fixture', project_name='toy_recovery_fixture' if recovery else 'toy_classification',
            runs=[RunPlan(name='one', base_config=RunConfig(output_dir=root / 'data/run', batch_size=128, epochs=1))])
        exp = service.create_experiment(plan)
        deps = WorkflowDependencies(service, FixturePlanner(plan),
            diagnosis_agent=DiagnosisAgent(FakeDiagnosisModel(), service, None, mode='fake'),
            recovery_service=RecoveryService(service, FixProposer(FakeFixModel())))
        config = {'configurable': {'thread_id': exp.id}}
        try:
            with sqlite_checkpointer(root / 'real_checkpoints.sqlite', application_db=root / 'real.sqlite') as saver:
                graph = build_experiment_graph(deps, saver, deferred_polling=True)
                graph.invoke({'experiment_id': exp.id, 'user_goal': exp.goal}, config)
                deadline = time.monotonic() + 45
                while graph.get_state(config).next == ('wait_for_poll',) and time.monotonic() < deadline:
                    time.sleep(.05)
                    graph.invoke(None, config)
                assert graph.get_state(config).values['workflow_status'] == 'SUCCEEDED'
            attempts = sorted(repo.list(Attempt), key=lambda a: a.attempt_no)
            assert len(attempts) == (2 if recovery else 1)
            if recovery:
                assert attempts[0].status == 'FAILED' and attempts[0].config_json['batch_size'] == 128
                assert attempts[1].config_json['batch_size'] == 64
            assert len(repo.list(Metric)) == 3
            assert all(a.backend == 'local' for a in attempts)
        finally:
            for job in backend.jobs.values():
                if job.process.poll() is None:
                    backend._terminate(job)
    factory.kw['bind'].dispose()
