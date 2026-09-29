import json
import shutil
from pathlib import Path
import pytest
from ml_experiment_agent.agent.diagnosis import DiagnosisAgent
from ml_experiment_agent.db.models import Run, Attempt, Metric, Diagnosis, AuditEvent
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer
from ml_experiment_agent.graph.dependencies import WorkflowDependencies, RunExecutionOptions
from ml_experiment_agent.graph.planning import FixturePlanner
from ml_experiment_agent.graph.workflow import build_experiment_graph
from ml_experiment_agent.llm.fix_proposer import FixProposer
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.services.hpc_bridge import HPCBridge
from ml_experiment_agent.services.recovery_service import RecoveryService
from ml_experiment_agent.testing import FakeDiagnosisModel, FakeFixModel


@pytest.fixture
def bridge(environment):
    root, registry, policy = environment
    factory = create_session_factory(root / 'bridge.sqlite')
    with factory() as session:
        repo = Repository(session)
        backend = HPCBridge(policy, repo, memory_unit='MB')  # fixture unit, no cluster assertion
        service = ExperimentService(repo, registry, policy, backend)
        definition = ExperimentDefinition(goal='fixture', runs=[RunPlan(name='one', parameters={'learning_rate': .001},
            base_config=RunConfig(output_dir=root / 'data/run', batch_size=128, epochs=1))])
        exp = service.create_experiment(definition)
        deps = WorkflowDependencies(service, FixturePlanner(definition),
            diagnosis_agent=DiagnosisAgent(FakeDiagnosisModel(), service, None, mode='fake'),
            recovery_service=RecoveryService(service, FixProposer(FakeFixModel())),
            default_execution_options=RunExecutionOptions(resources={'queue': '734090d!'}))
        with sqlite_checkpointer(root / 'checkpoint.sqlite', application_db=root / 'bridge.sqlite') as saver:
            graph = build_experiment_graph(deps, saver, deferred_polling=True)
            config = {'configurable': {'thread_id': exp.id}}
            yield root, service, backend, graph, exp, config
    factory.kw['bind'].dispose()


def export(bridge):
    root, service, backend, graph, exp, config = bridge
    graph.invoke({'experiment_id': exp.id, 'user_goal': exp.goal}, config)
    attempt = service.repo.latest_attempt(service.repo.list(Run, experiment_id=exp.id)[0].id)
    folder = root / 'data/exports' / exp.id / attempt.id
    return attempt, folder


def result(folder, *, failed=False):
    manifest = json.loads((folder / 'manifest.json').read_text())
    (folder / 'stdout.log').write_text('external fixture stdout')
    (folder / 'stderr.log').write_text('torch.cuda.OutOfMemoryError: CUDA out of memory' if failed else 'external stderr')
    (folder / 'exit.json').write_text(json.dumps({'attempt_id': manifest['attempt_id'], 'exit_code': 1 if failed else 0}))
    if not failed:
        (folder / 'checkpoint.pt').write_bytes(b'fixture checkpoint, metrics already supplied')
        (folder / 'metadata.json').write_text('{"fixture": true}')
        (folder / 'metrics.json').write_text(json.dumps({'accuracy': .8, 'macro_f1': .7, 'ece': .1}))


def test_valid_export_association_and_script(bridge):
    attempt, folder = export(bridge)
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['attempt_id'] == attempt.id and manifest['run_id'] == attempt.run_id
    assert manifest['experiment_id'] == bridge[4].id
    script = (folder / 'submit.sh').read_text()
    assert '#BSUB -q 734090d!' in script and 'rusage[mem=2048]' in script
    assert b'\r' not in (folder / 'submit.sh').read_bytes()
    assert str(bridge[0]) not in (folder / 'README.txt').read_text() + json.dumps(manifest)
    assert attempt.status == 'QUEUED' and bridge[3].get_state(bridge[5]).next == ('wait_for_poll',)
    bridge[3].invoke(None, bridge[5])
    assert len(bridge[1].repo.list(Attempt)) == 1


def test_success_import_metrics_logs_report_idempotent(bridge):
    attempt, folder = export(bridge)
    result(folder)
    backend, repo = bridge[2], bridge[1].repo
    assert backend.import_result(folder) == bridge[4].id
    assert backend.import_result(folder) == bridge[4].id
    bridge[3].invoke(None, bridge[5])
    state = bridge[3].get_state(bridge[5]).values
    assert state['workflow_status'] == 'SUCCEEDED' and '0.8' in state['final_report']
    assert len(repo.list(Metric)) == 3
    assert len(repo.list(AuditEvent, action='hpc_result_imported')) == 1
    output = Path(attempt.config_json['output_dir'])
    assert (output / 'stdout.log').read_text() == 'external fixture stdout'
    assert 'external stderr' in (output / 'job.log').read_text()


def test_failed_import_reuses_graph_diagnosis_recovery_next_bundle(bridge):
    old, folder = export(bridge)
    result(folder, failed=True)
    bridge[2].import_result(folder)
    bridge[3].invoke(None, bridge[5])
    repo = bridge[1].repo
    attempts = sorted(repo.list(Attempt), key=lambda a: a.attempt_no)
    assert len(attempts) == 2 and old.status == 'FAILED' and attempts[1].status == 'QUEUED'
    assert old.config_json['batch_size'] == 128 and attempts[1].config_json['batch_size'] == 64
    assert repo.list(Diagnosis, attempt_id=old.id)[0].error_type == 'CUDA_OOM'
    next_folder = folder.parent / attempts[1].id
    assert (next_folder / 'submit.sh').is_file()
    assert bridge[3].get_state(bridge[5]).next == ('wait_for_poll',)
    bridge[2].import_result(folder)  # replay of old failure must not recreate retry
    bridge[3].invoke(None, bridge[5])
    assert len(repo.list(Attempt)) == 2
    result(next_folder)
    bridge[2].import_result(next_folder)
    bridge[3].invoke(None, bridge[5])
    assert bridge[3].get_state(bridge[5]).values['workflow_status'] == 'SUCCEEDED'


@pytest.mark.parametrize('field,value', [('run_id', 'wrong'), ('experiment_id', 'wrong'), ('config', {})])
def test_wrong_mapping_rejected(bridge, field, value):
    attempt, folder = export(bridge)
    result(folder)
    data = json.loads((folder / 'manifest.json').read_text())
    data[field] = value
    (folder / 'manifest.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='association'):
        bridge[2].import_result(folder)
    assert attempt.status == 'QUEUED' and not bridge[1].repo.list(Metric)


def test_conflicting_reimport_rejected(bridge):
    _, folder = export(bridge)
    result(folder)
    bridge[2].import_result(folder)
    (folder / 'stdout.log').write_text('changed history')
    with pytest.raises(ValueError, match='Conflicting'):
        bridge[2].import_result(folder)


@pytest.mark.parametrize('name,content', [('exit.json', '{"attempt_id":"wrong","exit_code":0}'),
    ('metrics.json', '{"accuracy":5,"macro_f1":0.5,"ece":0.1}'), ('metadata.json', '[]')])
def test_invalid_result_rejected_before_writes(bridge, name, content):
    attempt, folder = export(bridge)
    result(folder)
    (folder / name).write_text(content)
    with pytest.raises(ValueError):
        bridge[2].import_result(folder)
    assert attempt.status == 'QUEUED'
    assert not (Path(attempt.config_json['output_dir']) / 'stdout.log').exists()


@pytest.mark.parametrize('changes', [{'entrypoint': 'evil.py'}, {'entrypoint': 'train.py;id'},
    {'config_path': Path('../outside.yaml')}, {'queue': 'bad'}, {'memory_gb': 100}])
def test_unsafe_export_rejected(bridge, changes):
    root, service, backend, _, exp, _ = bridge
    run = service.create_runs(exp.id)[0]
    attempt = service.create_attempt(run.id)
    job = service.build_job_spec(run.id, config=RunConfig.model_validate(attempt.config_json), queue='734090d!')
    with pytest.raises((ValueError, OSError)):
        backend.submit(job.model_copy(update=changes))
    assert not (root / 'data/exports').exists()


def test_unconfirmed_memory_blocks_export(bridge):
    _, service, backend, _, exp, _ = bridge
    run = service.create_runs(exp.id)[0]
    attempt = service.create_attempt(run.id)
    backend.memory_unit = None
    with pytest.raises(ValueError, match='confirmed memory'):
        service.submit_attempt(attempt.id, queue='734090d!')


def test_unsafe_config_cannot_export(bridge):
    _, service, _, _, exp, _ = bridge
    run = service.create_runs(exp.id)[0]
    attempt = service.create_attempt(run.id)
    (Path(attempt.config_json['output_dir']) / 'config.yaml').write_text('batch_size: 0\noutput_dir: .\n')
    with pytest.raises(ValueError):
        service.submit_attempt(attempt.id, queue='734090d!')


def test_missing_success_checkpoint_rejected(bridge):
    attempt, folder = export(bridge)
    result(folder)
    (folder / 'checkpoint.pt').unlink()
    with pytest.raises(ValueError, match='checkpoint'):
        bridge[2].import_result(folder)
    assert attempt.status == 'QUEUED'


def test_import_root_escape_rejected(bridge):
    with pytest.raises(ValueError, match='outside'):
        bridge[2].import_result(bridge[0] / '../outside')


def test_exported_runner_is_real_and_records_exit(bridge):
    import subprocess, sys
    attempt, folder = export(bridge)
    root = bridge[0]
    (root / 'toy_project/train.py').write_text('print("fixture training")\n')
    (root / 'toy_project/evaluate.py').write_text('import sys\nprint("fixture evaluation", file=sys.stderr)\n')
    run = subprocess.run([sys.executable, str(folder / 'run_bundle.py'), '--project-root', str(root)],
                         capture_output=True, text=True, check=True)
    assert json.loads((folder / 'exit.json').read_text()) == {'attempt_id': attempt.id, 'exit_code': 0}
    assert 'fixture training' in (folder / 'stdout.log').read_text()
    assert 'fixture evaluation' in (folder / 'stderr.log').read_text()
    duplicate = subprocess.run([sys.executable, str(folder / 'run_bundle.py'), '--project-root', str(root)],
                               capture_output=True, text=True)
    assert duplicate.returncode != 0 and 'already completed' in duplicate.stderr
