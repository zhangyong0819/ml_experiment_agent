"""Manual execution boundary: submit exports; status waits for explicit import.

No remote connection or scheduler command is executed by this adapter. The same
graph resumes after import, including its existing diagnosis/recovery nodes.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from ..backends.base import ComputeBackend
from ..backends.local import read_process_logs
from ..db.models import Attempt, Run, Experiment, now
from ..db.repository import Repository
from ..models.enums import JobStatus, ArtifactType
from ..models.experiment import MetricRecord, ArtifactRecord
from ..models.job import JobSpec, JobHandle
from .config_service import ConfigService
from .evaluation_service import EvaluationMetrics
from .policy_engine import PolicyEngine
from .artifact_service import ArtifactService
from .experiment_service import classify_error


class HPCBridge(ComputeBackend):
    """ComputeBackend semantics with human submission instead of automatic compute."""
    def __init__(self, policy: PolicyEngine, repository: Repository, *, memory_unit: str | None = None):
        if memory_unit not in (None, "MB", "GB"):
            raise ValueError("Memory unit must be explicitly confirmed as MB or GB")
        self.policy, self.repo, self.memory_unit = policy, repository, memory_unit

    def submit(self, job: JobSpec) -> JobHandle:
        decision = self.policy.validate(job, active_jobs=self.repo.active_count())
        if not decision.allowed:
            raise ValueError(decision.reason)
        if not job.queue:
            raise ValueError("HPC export requires an allowlisted queue")
        if self.memory_unit is None:
            raise ValueError("HPC training export requires a confirmed memory unit; not guessed")
        spec = self.policy.registry.get(job.project_name)
        if job.entrypoint != spec.train_entrypoint:
            raise ValueError("Only registered training entrypoints may be exported")
        evaluation = self.policy.validate(job.model_copy(update={"entrypoint": spec.eval_entrypoint}))
        if not evaluation.allowed:
            raise ValueError(evaluation.reason)
        output = self.policy.safe_path(job.output_dir)
        if job.config_path.resolve() != output / "config.yaml":
            raise ValueError("Unknown attempt config path")
        config = ConfigService().load(job.config_path)
        if config.output_dir.resolve() != output or config.num_workers + 1 > job.cpu_count:
            raise ValueError("Config does not match job output/resources")
        attempts = [a for a in self.repo.list(Attempt, status="CREATED") if a.config_json == config.model_dump(mode="json")]
        if len(attempts) != 1:
            raise ValueError("Export requires one matching registered CREATED Attempt")
        attempt = attempts[0]
        run = self.repo.get(Run, attempt.run_id)
        if self.repo.get(Experiment, run.experiment_id).plan_json["project_name"] != job.project_name:
            raise ValueError("Job project differs from the registered attempt")
        relative = spec.working_directory.relative_to(self.policy.root).as_posix()
        for value in (relative, job.entrypoint, spec.eval_entrypoint, spec.checkpoint_name, spec.metadata_name, spec.evaluation_name):
            if not re.fullmatch(r"[A-Za-z0-9_./-]+", value) or ".." in Path(value).parts or Path(value).is_absolute():
                raise ValueError("Unsafe registered bundle path")
        portable = config.model_dump(mode="json") | {"output_dir": "."}
        manifest = {"experiment_id": run.experiment_id, "run_id": run.id, "attempt_id": attempt.id,
            "project_name": job.project_name, "working_directory": relative,
            "entrypoint": job.entrypoint, "eval_entrypoint": spec.eval_entrypoint,
            "config": portable, "expected_logs": ["stdout.log", "stderr.log"],
            "expected_artifacts": [spec.checkpoint_name, spec.metadata_name, spec.evaluation_name],
            "queue": job.queue, "resources": {"cpu_count": job.cpu_count, "gpu_count": job.gpu_count,
            "memory_gb": job.memory_gb, "memory_unit": self.memory_unit, "timeout_minutes": job.timeout_minutes}}
        folder = self.policy.safe_path(self.policy.root / "data/exports" / run.experiment_id / attempt.id)
        existing = self.repo.latest_audit(run.experiment_id, "hpc_bundle_exported", run_id=run.id, attempt_id=attempt.id)
        if existing:
            if existing.output_json["manifest"] != manifest:
                raise ValueError("Export differs from immutable registered bundle")
            return JobHandle(job_id="hpc-" + attempt.id, backend="manual_hpc")
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "config.yaml", "submit.sh", "run_bundle.py", "README.txt"):
            if (folder / name).exists() or (folder / name).is_symlink():
                raise ValueError("Existing export needs operator reconciliation, never overwrite it")
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (folder / "config.yaml").write_text(json.dumps(portable, indent=2), encoding="utf-8")
        memory = math.ceil(job.memory_gb * (1024 if self.memory_unit == "MB" else 1))
        directives = ["#!/bin/sh", f"#BSUB -q {job.queue}", f"#BSUB -n {job.cpu_count}",
                      f"#BSUB -W {math.ceil(job.timeout_minutes)}", f'#BSUB -R "rusage[mem={memory}]"']
        if job.gpu_count:
            directives.append(f'#BSUB -gpu "num={job.gpu_count}:j_exclusive=yes"')
        directives += ["#BSUB -o scheduler.stdout.log", "#BSUB -e scheduler.stderr.log",
                       'exec python3 run_bundle.py --project-root "${ML_AGENT_PROJECT_ROOT:?Set project root}"']
        (folder / "submit.sh").write_text("\n".join(directives) + "\n", encoding="utf-8", newline="\n")
        shutil.copyfile(Path(__file__).with_name("hpc_bundle_runner.py"), folder / "run_bundle.py")
        (folder / "README.txt").write_text(
            "1. Upload this directory beneath your HPC project root; project code/dependencies must already exist.\n"
            "2. Login manually, set ML_AGENT_PROJECT_ROOT to that project root, and cd into this run directory.\n"
            "3. Run bsub < submit.sh (from this directory). Do not submit this attempt twice.\n"
            "4. Wait for completion, then download the entire run directory.\n"
            "5. Run python scripts/import_hpc_result.py <downloaded-directory> --memory-unit <confirmed-unit> locally.\n"
            "Python 3 and the registered project's dependencies must be available remotely.\n", encoding="utf-8")
        self.repo.audit(run.experiment_id, "hpc_bundle_exported", run_id=run.id, attempt_id=attempt.id,
                        outputs={"manifest": manifest, "bundle": str(folder)})
        self.repo.commit()
        return JobHandle(job_id="hpc-" + attempt.id, backend="manual_hpc")

    def _attempt(self, job_id: str) -> Attempt:
        if not job_id.startswith("hpc-"):
            raise KeyError("Unknown external job")
        attempt = self.repo.get(Attempt, job_id[4:])
        if attempt.backend != "manual_hpc" or attempt.job_id != job_id:
            raise KeyError("Unknown external attempt ownership")
        return attempt

    def status(self, job_id: str) -> JobStatus:
        return JobStatus(self._attempt(job_id).status)

    def logs(self, job_id: str, tail=200):
        attempt = self._attempt(job_id)
        return read_process_logs(self.policy, self.policy.safe_path(Path(attempt.config_json["output_dir"])), job_id, tail)

    def cancel(self, job_id: str) -> None:
        self._attempt(job_id)
        raise ValueError("External jobs must be cancelled by the operator; no remote command is available")

    def import_result(self, directory: Path) -> str:
        """Validate all files/associations before writing; identical imports are no-ops."""
        directory = self.policy.safe_path(directory)
        def read(name: str, required=True) -> bytes | None:
            path = (directory / name).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Result file escapes download directory")
            if not path.is_file() and not required:
                return None
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Result file exceeds the 16 MiB toy-project bound")
            return path.read_bytes()
        manifest = json.loads(read("manifest.json"))
        attempt = self._attempt("hpc-" + manifest["attempt_id"])
        run = self.repo.get(Run, attempt.run_id)
        record = self.repo.latest_audit(run.experiment_id, "hpc_bundle_exported", run_id=run.id, attempt_id=attempt.id)
        if record is None or record.output_json["manifest"] != manifest:
            raise ValueError("Manifest/run/attempt/config association mismatch")
        if json.loads(read("config.yaml")) != manifest["config"]:
            raise ValueError("Downloaded config differs from the registered export")
        exit_info = json.loads(read("exit.json"))
        if set(exit_info) != {"attempt_id", "exit_code"} or exit_info["attempt_id"] != attempt.id or type(exit_info["exit_code"]) is not int:
            raise ValueError("Reliable attempt-scoped integer exit information required")
        spec = self.policy.registry.get(manifest["project_name"])
        blobs = {name: read(name) for name in manifest["expected_logs"]}
        for name in manifest["expected_artifacts"]:
            value = read(name, required=False)
            if value is not None:
                blobs[name] = value
        success = exit_info["exit_code"] == 0
        if success and spec.checkpoint_name not in blobs:
            raise ValueError("Successful import is missing its registered checkpoint")
        metrics = None
        if success and spec.evaluation_name in blobs:
            metrics = EvaluationMetrics.model_validate(json.loads(blobs[spec.evaluation_name]))
            if set(metrics.model_dump()) != set(spec.metric_names):
                raise ValueError("Metric schema differs from project")
        if spec.metadata_name in blobs and not isinstance(json.loads(blobs[spec.metadata_name]), dict):
            raise ValueError("Artifact metadata must be a JSON object")
        digest = hashlib.sha256(json.dumps(exit_info, sort_keys=True).encode())
        for name, content in sorted(blobs.items()):
            digest.update(name.encode() + b"\0" + content)
        fingerprint = digest.hexdigest()
        imported = self.repo.latest_audit(run.experiment_id, "hpc_result_imported", run_id=run.id, attempt_id=attempt.id)
        if imported:
            if imported.output_json["fingerprint"] != fingerprint:
                raise ValueError("Conflicting reimport; previous attempt evidence is immutable")
            return run.experiment_id
        if attempt.status not in ("QUEUED", "RUNNING"):
            raise ValueError("Only an outstanding exported attempt can receive results")
        output = self.policy.safe_path(Path(attempt.config_json["output_dir"]))
        destinations = {name: self.policy.safe_path(output / name) for name in blobs}
        if any(not path.is_relative_to(output) for path in destinations.values()):
            raise ValueError("Import destination escapes attempt")
        for name, content in blobs.items():
            destinations[name].write_bytes(content)
        attempt.status = "SUCCEEDED" if success else "FAILED"
        attempt.ended_at = now()
        if not success:
            log = self.logs(attempt.job_id).text
            attempt.error_type = classify_error(log).value
            attempt.error_message = log[-2000:]
        if metrics:
            for name, value in metrics.model_dump().items():
                self.repo.save_metric(**MetricRecord(attempt_id=attempt.id, metric_name=name, metric_value=value).model_dump())
        kinds = {"stdout.log": ArtifactType.LOG, "stderr.log": ArtifactType.LOG,
                 spec.checkpoint_name: ArtifactType.CHECKPOINT, spec.metadata_name: ArtifactType.METADATA,
                 spec.evaluation_name: ArtifactType.EVALUATION}
        for name, path in destinations.items():
            ArtifactService(self.repo, self.policy).record(ArtifactRecord(attempt_id=attempt.id, artifact_type=kinds[name], path=path), output)
        self.repo.sync_status(attempt)
        self.repo.audit(run.experiment_id, "hpc_result_imported", run_id=run.id, attempt_id=attempt.id,
                        outputs={"fingerprint": fingerprint, "exit_code": exit_info["exit_code"]})
        self.repo.commit()
        return run.experiment_id
