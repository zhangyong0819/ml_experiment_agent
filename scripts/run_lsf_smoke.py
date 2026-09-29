"""Validate the existing LSFBackend with one CPU-only, standard-library probe."""
import argparse
import json
import os
from pathlib import Path
import time
from ml_experiment_agent.backends.cluster import ClusterConfig
from ml_experiment_agent.backends.lsf import LSFBackend
from ml_experiment_agent.db.models import Attempt
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.models.project import ProjectRegistry
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.services.policy_engine import PolicyEngine


def load_connection_environment(path: Path) -> None:
    """Read only connection settings; ignore credentials and never log file content."""
    allowed = {'LSF_SSH_HOST', 'LSF_SSH_PORT', 'LSF_SSH_USERNAME', 'LSF_SSH_HOST_ALIAS',
               'LSF_REMOTE_PROJECT_ROOT', 'ENABLE_REAL_LSF', 'LSF_MEMORY_UNIT', 'LSF_QUEUE'}
    if path.is_file():
        with path.open(encoding='utf-8-sig') as handle:
            for line in handle:
                key = line.split('=', 1)[0].strip()
                if key in allowed and '=' in line:
                    value = line.split('=', 1)[1].strip().strip('\"\'')
                    os.environ.setdefault(key, value)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=root / '.env')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--attempt-id', help='Continue an owned existing submission, never create a replacement')
    parser.add_argument('--wait-minutes', type=float, default=5)
    args = parser.parse_args()
    load_connection_environment(args.env_file)
    if os.getenv('ENABLE_REAL_LSF', 'false').lower() != 'true':
        raise SystemExit('BLOCKED: ENABLE_REAL_LSF=true is required; no SSH or submission attempted.')
    cluster = ClusterConfig.from_environment()
    cluster.interactive_auth = args.interactive
    registry = ProjectRegistry(root / 'project_specs', root)
    policy = PolicyEngine.load(root / 'configs/policy.yaml', root, registry)
    factory = create_session_factory(root / 'data/lsf_smoke.sqlite')
    evidence = {'ssh_success': False, 'queue': cluster.queue, 'job_id': None,
                'observed_statuses': [], 'terminal': None, 'artifact_read': False,
                'stdout_verified': False, 'stderr_verified': False, 'result': 'FAIL'}
    try:
        with factory() as session:
            repo = Repository(session)
            backend = LSFBackend(policy, repo, cluster)
            backend.check_connection(smoke=True)
            evidence['ssh_success'] = True
            print('SSH and registered smoke entrypoint ready', flush=True)
            if args.check_only:
                return
            service = ExperimentService(repo, registry, policy, backend)
            if args.attempt_id:
                attempt = repo.get(Attempt, args.attempt_id)
                if not attempt.job_id:
                    raise ValueError('Attempt has no committed job ID; operator reconciliation required')
            else:
                exp = service.create_experiment(ExperimentDefinition(goal='Minimal scheduler probe', project_name='lsf_smoke'))
                run = service.create_runs(exp.id, [RunPlan(name='lsf_probe', base_config=RunConfig(output_dir=root / 'data/lsf_smoke_runs'))])[0]
                attempt = service.create_attempt(run.id)
                print('Attempt ID:', attempt.id, flush=True)
                service.submit_attempt(attempt.id, queue=cluster.queue, gpu_count=0, cpu_count=1,
                                       memory_gb=1, timeout_minutes=1)
            evidence['job_id'] = attempt.job_id
            print('Submitted queue:', cluster.queue, 'Job ID:', attempt.job_id, flush=True)
            # Reconstruct the backend to verify application-owned jobs survive restart.
            service.backend = LSFBackend(policy, repo, cluster)
            deadline = time.monotonic() + max(0, args.wait_minutes) * 60
            terminal = {'SUCCEEDED', 'FAILED', 'CANCELLED', 'TIMED_OUT'}
            while time.monotonic() <= deadline:
                service.refresh_attempt_status(attempt.id)
                evidence['observed_statuses'].append(attempt.status)
                print('Observed:', attempt.status, flush=True)
                if attempt.status in terminal:
                    break
                time.sleep(cluster.poll_interval_seconds)
            evidence['terminal'] = attempt.status if attempt.status in terminal else None
            log = service.collect_logs(attempt.id)
            evidence['stdout_verified'] = 'hostname:' in log.text and 'done' in log.text
            evidence['stderr_verified'] = 'smoke stderr marker' in log.text
            print(log.text, flush=True)
            if attempt.status == 'SUCCEEDED':
                artifact = service.backend.smoke_artifact(attempt.job_id)
                evidence['artifact_read'] = artifact == 'LSF smoke succeeded\n'
                output = policy.safe_path(Path(attempt.config_json['output_dir']))
                target = policy.safe_path(output / 'smoke_result.txt')
                if not target.is_relative_to(output):
                    raise ValueError('Local smoke artifact escapes attempt directory')
                target.write_text(artifact, encoding='utf-8')
            if (evidence['terminal'] == 'SUCCEEDED' and evidence['artifact_read']
                    and evidence['stdout_verified'] and evidence['stderr_verified']):
                evidence['result'] = 'PASS'
            print(json.dumps(evidence, indent=2), flush=True)
            if evidence['result'] != 'PASS':
                raise SystemExit('LSF smoke FAIL/incomplete; inspect local evidence. No replacement job was submitted.')
    finally:
        directory = root / 'data'
        directory.mkdir(exist_ok=True)
        (directory / 'lsf_smoke_validation.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        factory.kw['bind'].dispose()


if __name__ == '__main__':
    main()
