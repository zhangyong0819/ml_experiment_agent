"""Real CPU processes owned by registered attempts, within one Agent process."""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4
from .base import ComputeBackend
from ..db.models import Attempt, Run, Experiment
from ..db.repository import Repository
from ..models.enums import JobStatus, Scenario
from ..models.job import JobSpec, JobHandle, JobLog
from ..services.config_service import ConfigService
from ..services.policy_engine import PolicyEngine


@dataclass
class LocalJob:
    process: subprocess.Popen
    attempt_id: str
    output: Path
    started: float
    timeout: float
    state: JobStatus = JobStatus.RUNNING


class LocalBackend(ComputeBackend):
    """No shell and no PID reattachment. Process handles are deliberately ephemeral."""
    def __init__(self, policy: PolicyEngine, repository: Repository, *,
                 jobs: dict[str, LocalJob] | None = None):
        self.policy, self.repo = policy, repository
        # One application's handles can outlive invocation-scoped repositories.
        # This is still process-local ownership, not restart/PID reattachment.
        self.jobs: dict[str, LocalJob] = jobs if jobs is not None else {}

    def submit(self, job: JobSpec) -> JobHandle:
        decision = self.policy.validate(job, active_jobs=self.repo.active_count())
        if not decision.allowed:
            raise ValueError(decision.reason)
        config = ConfigService().load(self.policy.safe_path(job.config_path))
        output = self.policy.safe_path(job.output_dir)
        if config.output_dir.resolve() != output or job.config_path.resolve() != output / "config.yaml":
            raise ValueError("Attempt config path/output mismatch")
        if job.gpu_count or config.num_workers or job.scenario != Scenario.SUCCESS:
            raise ValueError("LocalBackend supports CPU, zero data workers, and real execution only")
        spec = self.policy.registry.get(job.project_name)
        if job.entrypoint != spec.train_entrypoint:
            raise ValueError("Only the registered training entrypoint is executable")
        matches = [a for a in self.repo.list(Attempt, status="CREATED") if a.config_json == config.model_dump(mode="json")]
        if len(matches) != 1:
            raise ValueError("Exactly one registered CREATED attempt is required")
        attempt = matches[0]
        run = self.repo.get(Run, attempt.run_id)
        if self.repo.get(Experiment, run.experiment_id).plan_json["project_name"] != job.project_name:
            raise ValueError("Job project differs from the registered attempt")
        if self.repo.latest_audit(run.experiment_id, "local_submission_intent", run_id=run.id, attempt_id=attempt.id):
            raise OSError("Local submission already started; do not duplicate an uncertain process")
        paths = [self.policy.safe_path(output / name) for name in ("stdout.log", "stderr.log")]
        if any(not p.is_relative_to(output) for p in paths):
            raise ValueError("Log path escapes attempt directory")
        self.repo.audit(run.experiment_id, "local_submission_intent", run_id=run.id, attempt_id=attempt.id)
        self.repo.commit()
        with paths[0].open("wb") as stdout, paths[1].open("wb") as stderr:
            process = subprocess.Popen([sys.executable, str(self.policy.safe_path(job.working_directory / job.entrypoint)),
                "--config", str(job.config_path.resolve())], cwd=job.working_directory,
                stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, shell=False)
        identifier = "local-" + uuid4().hex
        self.jobs[identifier] = LocalJob(process, attempt.id, output, time.monotonic(), job.timeout_minutes * 60)
        return JobHandle(job_id=identifier, backend="local", status=JobStatus.RUNNING)

    def _get(self, identifier: str) -> LocalJob:
        if identifier not in self.jobs:
            raise KeyError("Unknown local job; active processes cannot be reattached after restart")
        job = self.jobs[identifier]
        attempt = self.repo.get(Attempt, job.attempt_id)
        if attempt.backend != "local" or attempt.job_id != identifier:
            raise KeyError("Local job is not registered to this application attempt")
        return job

    def status(self, job_id: str) -> JobStatus:
        job = self._get(job_id)
        if job.state != JobStatus.RUNNING:
            return job.state
        code = job.process.poll()
        if code is not None:
            job.state = JobStatus.SUCCEEDED if code == 0 else JobStatus.FAILED
        elif time.monotonic() - job.started >= job.timeout:
            self._terminate(job)
            job.state = JobStatus.TIMED_OUT
        return job.state

    @staticmethod
    def _terminate(job: LocalJob) -> None:
        job.process.terminate()
        try:
            job.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            job.process.kill()
            job.process.wait(timeout=3)

    def logs(self, job_id: str, tail: int = 200) -> JobLog:
        job = self._get(job_id)
        return read_process_logs(self.policy, job.output, job_id, tail)

    def cancel(self, job_id: str) -> None:
        job = self._get(job_id)
        if self.status(job_id) == JobStatus.RUNNING:
            self._terminate(job)
            job.state = JobStatus.CANCELLED


def read_process_logs(policy: PolicyEngine, output: Path, job_id: str, tail: int) -> JobLog:
    """Bound both file reads and returned lines; never checkpoint full output."""
    if type(tail) is not int or not 1 <= tail <= 200:
        raise ValueError("tail must be 1..200")
    lines = []
    for name in ("stdout.log", "stderr.log"):
        path = policy.safe_path(output / name)
        if not path.is_relative_to(output):
            raise ValueError("Log path escapes attempt")
        if path.is_file():
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 65536))
                lines += [f"[{name}]"] + handle.read(65536).decode("utf-8", errors="replace").splitlines()
    return JobLog(job_id=job_id, text="\n".join(lines[-tail:])[-32000:])
