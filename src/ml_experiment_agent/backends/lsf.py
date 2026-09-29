"""Single-cluster SSH backend with durable application-owned job records."""
import base64
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
from .base import ComputeBackend
from .cluster import ClusterConfig, remote_path
from ..db.models import Attempt, Run, AuditEvent
from ..db.repository import Repository
from ..models.enums import JobStatus
from ..models.job import JobSpec, JobHandle, JobLog
from ..services.config_service import ConfigService
from ..services.policy_engine import PolicyEngine


class LSFBackend(ComputeBackend):
    """Only registered attempts can submit, inspect, transfer artifacts, or cancel.

    The database audit record survives a new backend/process. An ambiguous submit
    is deliberately not retried: the workflow's existing submission intent blocks
    replay until an operator reconciles it with the scheduler.
    """

    def __init__(self, policy: PolicyEngine, repository: Repository, cluster: ClusterConfig):
        self.policy, self.repo, self.cluster = policy, repository, cluster

    def _remote(self, operation: str, **payload) -> dict:
        if not self.cluster.enabled or os.getenv("ENABLE_REAL_LSF", "false").lower() != "true":
            raise ValueError("Real LSF disabled: require cluster.enabled and ENABLE_REAL_LSF=true")
        source = Path(__file__).with_name("remote_operations.py").read_text(encoding="utf-8")
        # The command string is fixed project code. Dynamic data travels via stdin,
        # never through the remote shell, ssh options, or a user-supplied command.
        command = "python3 -c " + shlex.quote(source)
        try:
            result = subprocess.run(self.cluster.ssh_arguments() + [command],
                input=json.dumps({"operation": operation, "root": self.cluster.remote_project_root, **payload}),
                stdout=subprocess.PIPE,
                stderr=None if self.cluster.interactive_auth else subprocess.PIPE,
                text=True, timeout=180 if self.cluster.interactive_auth else 60, check=False)
        except subprocess.TimeoutExpired as exc:
            raise OSError("SSH timed out; submission outcome may be uncertain") from exc
        if result.returncode:
            raise OSError(f"Remote {operation} failed: {(result.stderr or 'see local SSH terminal')[-1000:]}")
        try:
            response = json.loads(result.stdout)
        except ValueError as exc:
            raise OSError("Invalid remote response; submission outcome may be uncertain") from exc
        if not isinstance(response, dict):
            raise OSError("Invalid remote response object")
        return response

    def check_connection(self, *, smoke: bool = False) -> dict:
        return self._remote("check", smoke=smoke)

    def smoke_artifact(self, job_id: str) -> str:
        """Read the single fixed probe artifact from an owned successful job."""
        record = self._record(job_id)
        attempt = self.repo.get(Attempt, record.attempt_id)
        if attempt.status != "SUCCEEDED":
            raise ValueError("Smoke artifact requires a successful persisted attempt")
        result = self._remote("artifact", output=record.output_json["output"], name="smoke_result.txt")
        content = base64.b64decode(result["data"], validate=True)
        if len(content) > 4096:
            raise ValueError("Smoke artifact too large")
        return content.decode("utf-8")

    def _relative(self, path: Path) -> str:
        relative = self.policy.safe_path(path).relative_to(self.policy.root).as_posix()
        remote_path(self.cluster.remote_project_root + "/" + relative)
        return relative

    def submit(self, job: JobSpec) -> JobHandle:
        decision = self.policy.validate(job)
        if not decision.allowed:
            raise ValueError(decision.reason)
        if not job.queue or job.queue not in self.cluster.allowed_queues:
            raise ValueError("An explicitly allowlisted JobSpec queue is required")
        spec = self.policy.registry.get(job.project_name)
        smoke = spec.project_name == "lsf_smoke" and job.entrypoint == "lsf_smoke.py"
        if self.cluster.memory_unit is None and not smoke:
            raise ValueError("Confirm the cluster LSF memory request unit before submission")
        if job.entrypoint != spec.train_entrypoint:
            raise ValueError("Only registered training jobs can be submitted")
        config = ConfigService().load(self.policy.safe_path(job.config_path))
        if config.output_dir.resolve() != job.output_dir.resolve() or config.num_workers > job.cpu_count:
            raise ValueError("Config output/resources differ from JobSpec")
        matches = [a for a in self.repo.list(Attempt, status="CREATED")
                   if a.config_json == config.model_dump(mode="json")]
        if len(matches) != 1:
            raise ValueError("Submission requires one registered CREATED Attempt with matching config")
        attempt = matches[0]
        run = self.repo.get(Run, attempt.run_id)
        if self.repo.latest_audit(run.experiment_id, "lsf_job_registered", run_id=run.id, attempt_id=attempt.id):
            raise ValueError("Attempt already submitted; reconcile its registered job")
        output = self._relative(job.output_dir)
        working = self._relative(job.working_directory)
        entrypoint = self._relative(job.working_directory / job.entrypoint)
        if self.repo.latest_audit(run.experiment_id, "lsf_submission_intent", run_id=run.id, attempt_id=attempt.id):
            raise OSError("Previous LSF submission may have succeeded; reconciliation required")
        self.repo.audit(run.experiment_id, "lsf_submission_intent", run_id=run.id, attempt_id=attempt.id)
        self.repo.commit()
        response = self._remote("submit", working=working,
            entrypoint=entrypoint, output=output,
            config=config.model_dump(mode="json"), queue=job.queue, cpu=job.cpu_count,
            gpu=job.gpu_count, timeout=job.timeout_minutes,
            memory=None if smoke else math.ceil(job.memory_gb * (1024 if self.cluster.memory_unit == "MB" else 1)))
        match = re.fullmatch(r"\s*Job <([1-9][0-9]*)> is submitted to queue <([^<>\r\n]+)>\.?\s*", response.get("stdout", ""))
        if response.get("returncode") != 0 or not match or match[2] != job.queue:
            raise OSError("bsub response not recognized; do not resubmit until reconciled")
        job_id = match[1]
        self.repo.audit(run.experiment_id, "lsf_job_registered", run_id=run.id, attempt_id=attempt.id,
            outputs={"job_id": job_id, "output": output, "connection": self.cluster.connection_identity,
                     "root": self.cluster.remote_project_root})
        self.repo.commit()
        return JobHandle(job_id=job_id, backend="lsf")

    def _record(self, job_id: str) -> AuditEvent:
        if not re.fullmatch(r"[1-9][0-9]*", job_id):
            raise KeyError("Unknown LSF job")
        records = [row for row in self.repo.list(AuditEvent, action="lsf_job_registered")
                   if row.output_json.get("job_id") == job_id
                   and row.output_json.get("connection") == self.cluster.connection_identity
                   and row.output_json.get("root") == self.cluster.remote_project_root]
        if len(records) != 1:
            raise KeyError("Unknown or ambiguous application-owned LSF job")
        attempt = self.repo.get(Attempt, records[0].attempt_id)
        if attempt.backend != "lsf" or attempt.job_id != job_id:
            raise KeyError("LSF submission awaits application reconciliation")
        return records[0]

    def status(self, job_id: str) -> JobStatus:
        self._record(job_id)
        response = self._remote("status", job_id=job_id)
        status = response.get("stdout", "").strip()
        mapping = {"PEND": JobStatus.QUEUED, "PROV": JobStatus.QUEUED, "PSUSP": JobStatus.QUEUED,
                   "RUN": JobStatus.RUNNING, "USUSP": JobStatus.RUNNING, "SSUSP": JobStatus.RUNNING,
                   "DONE": JobStatus.SUCCEEDED, "EXIT": JobStatus.FAILED}
        if response.get("returncode") != 0 or status not in mapping:
            raise OSError("LSF status unavailable/unknown; operator reconciliation required")
        return mapping[status]

    def logs(self, job_id: str, tail: int = 200) -> JobLog:
        record = self._record(job_id)
        if type(tail) is not int or not 1 <= tail <= 200:
            raise ValueError("tail must be 1..200")
        result = self._remote("logs", output=record.output_json["output"], tail=tail)
        return JobLog(job_id=job_id, text=result["text"][-32000:])

    def cancel(self, job_id: str) -> None:
        record = self._record(job_id)
        if self.status(job_id) in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMED_OUT):
            return
        result = self._remote("cancel", job_id=job_id)
        if result.get("returncode") != 0:
            raise OSError("bkill failed")
        self.repo.audit(record.experiment_id, "lsf_cancel_requested", run_id=record.run_id, attempt_id=record.attempt_id)
        self.repo.commit()

    def prepare(self, attempt_id: str) -> None:
        """ArtifactPreparer bridge: download actual outputs, never simulate training.

        Existing EvaluationService evaluates the downloaded checkpoint on local CPU.
        """
        attempt = self.repo.get(Attempt, attempt_id)
        if attempt.status != "SUCCEEDED":
            raise ValueError("Only successful attempts have training artifacts")
        record = self._record(attempt.job_id)
        output = self.policy.safe_path(Path(attempt.config_json["output_dir"]))
        for name in ("checkpoint.pt", "metadata.json"):
            target = self.policy.safe_path(output / name)
            if not target.is_relative_to(output):
                raise ValueError("Artifact path escapes attempt")
            result = self._remote("artifact", output=record.output_json["output"], name=name)
            content = base64.b64decode(result["data"], validate=True)
            target.write_bytes(content)
        self.repo.audit(record.experiment_id, "remote_artifacts_downloaded", run_id=record.run_id, attempt_id=attempt.id)
        self.repo.commit()
