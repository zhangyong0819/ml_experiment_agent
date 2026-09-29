from dataclasses import dataclass
from .base import ComputeBackend
from ..models.job import JobSpec, JobHandle, JobLog
from ..models.enums import JobStatus, Scenario
from ..services.config_service import ConfigService
from ..services.policy_engine import PolicyEngine

FIXTURES = {
    Scenario.SUCCESS: "Training complete (simulated). Real artifacts are produced by EvaluationService.",
    Scenario.CUDA_OOM: "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB.",
    Scenario.CONFIG_RUNTIME_ERROR: "ValueError: config error: incompatible model dimensions",
    Scenario.RUNTIME_ERROR: "RuntimeError: loss became non-finite during forward pass",
    Scenario.TIMEOUT: "TIME LIMIT: scheduler terminated job after timeout",
    Scenario.PREEMPTED: "PREEMPTED: scheduler reclaimed node for higher priority job",
    Scenario.IMPORT_ERROR: "ImportError: No module named 'custom_extension'",
    Scenario.PATH_ERROR: "FileNotFoundError: [Errno 2] No such file or directory: 'dataset.bin'",
}


@dataclass
class MockJob:
    spec: JobSpec
    state: JobStatus = JobStatus.QUEUED
    queries: int = 0


class MockBackend(ComputeBackend):
    """Poll-driven simulator. Does not execute training or fabricate artifacts."""

    def __init__(self, policy: PolicyEngine):
        self.policy = policy
        self.jobs: dict[str, MockJob] = {}

    def submit(self, job: JobSpec) -> JobHandle:
        active = sum(j.state in (JobStatus.QUEUED, JobStatus.RUNNING) for j in self.jobs.values())
        decision = self.policy.validate(job, active_jobs=active)
        if not decision.allowed:
            raise ValueError(decision.reason)
        config = ConfigService().load(self.policy.safe_path(job.config_path))
        if config.num_workers + 1 > job.cpu_count:
            raise ValueError("cpu_count does not cover training and data workers")
        if self.policy.safe_path(config.output_dir) != self.policy.safe_path(job.output_dir):
            raise ValueError("Config output_dir does not match JobSpec")
        if job.scenario in (Scenario.INVALID_CONFIG, Scenario.POLICY_VIOLATION):
            raise ValueError("Validation scenarios must be rejected before submit")
        job_id = f"mock-{len(self.jobs) + 1:06d}"
        self.jobs[job_id] = MockJob(job.model_copy(deep=True))
        return JobHandle(job_id=job_id)

    def _get(self, job_id: str) -> MockJob:
        if job_id not in self.jobs:
            raise KeyError(f"Unknown job_id: {job_id}")
        return self.jobs[job_id]

    def status(self, job_id: str) -> JobStatus:
        job = self._get(job_id)
        if job.state not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return job.state
        terminal = JobStatus.SUCCEEDED if job.spec.scenario == Scenario.SUCCESS else (
            JobStatus.TIMED_OUT if job.spec.scenario == Scenario.TIMEOUT else JobStatus.FAILED)
        job.state = (JobStatus.QUEUED, JobStatus.RUNNING, terminal)[min(job.queries, 2)]
        job.queries += 1
        return job.state

    def logs(self, job_id: str, tail: int = 200) -> JobLog:
        job = self._get(job_id)
        if tail < 0:
            raise ValueError("tail must be nonnegative")
        lines = [f"Job {job_id} accepted by mock scheduler: QUEUED"]
        if job.queries >= 2:
            lines.append("RUNNING: initialized PyTorch training worker")
        if job.state == JobStatus.CANCELLED:
            lines.append("CANCELLED by user")
        elif job.state not in (JobStatus.QUEUED, JobStatus.RUNNING):
            lines.append(FIXTURES[job.spec.scenario])
        return JobLog(job_id=job_id, text="\n".join(lines[-tail:]) if tail else "")

    def cancel(self, job_id: str) -> None:
        job = self._get(job_id)
        if job.state in (JobStatus.QUEUED, JobStatus.RUNNING):
            job.state = JobStatus.CANCELLED
