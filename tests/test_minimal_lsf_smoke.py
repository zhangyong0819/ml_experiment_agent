import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from ml_experiment_agent.backends.cluster import ClusterConfig
from ml_experiment_agent.backends.lsf import LSFBackend
from ml_experiment_agent.backends.remote_operations import execute
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.services.experiment_service import ExperimentService
from test_lsf import lsf, submit

ROOT = Path(__file__).resolve().parents[1]


def test_probe_is_standard_library_only(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text(json.dumps({'output_dir': str(tmp_path)}))
    result = subprocess.run([sys.executable, '-S', str(ROOT / 'toy_project/lsf_smoke.py'), '--config', str(config)],
                            capture_output=True, text=True, check=True)
    assert 'hostname:' in result.stdout and 'done' in result.stdout
    assert 'smoke stderr marker' in result.stderr
    assert (tmp_path / 'smoke_result.txt').read_text() == 'LSF smoke succeeded\n'


def test_smoke_uses_backend_without_memory_unit(environment, monkeypatch):
    root, registry, policy = environment
    factory = create_session_factory(root / 'probe.sqlite')
    with factory() as session:
        repo = Repository(session)
        backend = LSFBackend(policy, repo, ClusterConfig(ssh_host_alias='school', remote_project_root='/work/probe'))
        calls = []
        def remote(operation, **values):
            calls.append((operation, values))
            return {'returncode': 0, 'stdout': 'Job <456> is submitted to queue <734090d!>.'}
        monkeypatch.setattr(backend, '_remote', remote)
        service = ExperimentService(repo, registry, policy, backend)
        exp = service.create_experiment(ExperimentDefinition(goal='probe', project_name='lsf_smoke'))
        run = service.create_runs(exp.id, [RunPlan(name='probe', base_config=RunConfig(output_dir=root / 'data/probe'))])[0]
        attempt = service.create_attempt(run.id)
        service.submit_attempt(attempt.id, queue='734090d!', gpu_count=0, cpu_count=1, timeout_minutes=1)
        assert calls[0][1]['memory'] is None and calls[0][1]['gpu'] == 0
        assert calls[0][1]['entrypoint'] == 'toy_project/lsf_smoke.py'
        assert LSFBackend(policy, repo, backend.cluster)._record('456').attempt_id == attempt.id
    factory.kw['bind'].dispose()


def test_training_still_requires_memory_unit(lsf):
    service, backend, attempt, calls = lsf
    backend.cluster.memory_unit = None
    with pytest.raises(ValueError, match='memory request unit'):
        service.submit_attempt(attempt.id, queue='734090d!')
    assert not calls


def test_remote_smoke_command_omits_memory_and_gpu(tmp_path, monkeypatch):
    captured = []
    def run(args, **kw):
        captured.append(args)
        return SimpleNamespace(returncode=0, stdout='submitted', stderr='')
    monkeypatch.setattr('ml_experiment_agent.backends.remote_operations.subprocess.run', run)
    execute({'operation': 'submit', 'root': str(tmp_path), 'working': 'toy_project',
             'entrypoint': 'toy_project/lsf_smoke.py', 'output': 'data/probe', 'config': {},
             'queue': '734090d!', 'cpu': 1, 'gpu': 0, 'timeout': 1, 'memory': None})
    assert '-R' not in captured[0] and '-M' not in captured[0] and '-gpu' not in captured[0]
    assert sys.executable in captured[0]


def test_remote_training_cannot_omit_memory(tmp_path):
    with pytest.raises(ValueError, match='default memory'):
        execute({'operation': 'submit', 'root': str(tmp_path), 'working': 'toy_project',
                 'entrypoint': 'toy_project/train.py', 'output': 'data/probe', 'config': {},
                 'queue': '734090d!', 'cpu': 1, 'gpu': 0, 'timeout': 1, 'memory': None})


def test_smoke_artifact_requires_owned_terminal_attempt(lsf, monkeypatch):
    backend, attempt, _ = submit(lsf)
    with pytest.raises(ValueError):
        backend.smoke_artifact('123')
    with pytest.raises(KeyError):
        backend.smoke_artifact('999')
    attempt.status = 'SUCCEEDED'
    backend.repo.commit()
    monkeypatch.setattr(backend, '_remote', lambda *a, **kw: {'data': base64.b64encode(b'LSF smoke succeeded\n').decode()})
    assert backend.smoke_artifact('123') == 'LSF smoke succeeded\n'


def test_interactive_mode_never_accepts_new_host_key():
    config = ClusterConfig(ssh_host_alias='school', remote_project_root='/work/probe', interactive_auth=True)
    assert 'StrictHostKeyChecking=yes' in config.ssh_arguments()
    assert 'StrictHostKeyChecking=ask' not in config.ssh_arguments()


def test_dotenv_loads_only_connection_settings(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('smoke_script', ROOT / 'scripts/run_lsf_smoke.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / '.env'
    path.write_text('LSF_SSH_HOST=fixture.example\nPASSWORD=not-a-real-password\nDEEPSEEK_API_KEY=not-a-real-key\n')
    for key in ('LSF_SSH_HOST', 'PASSWORD', 'DEEPSEEK_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    module.load_connection_environment(path)
    import os
    assert os.environ['LSF_SSH_HOST'] == 'fixture.example'
    assert 'PASSWORD' not in os.environ and 'DEEPSEEK_API_KEY' not in os.environ
    monkeypatch.delenv('LSF_SSH_HOST')
