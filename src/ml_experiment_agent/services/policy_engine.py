from pathlib import Path
import yaml
from ..models.job import JobSpec
from ..models.policy import PolicyConfig, PolicyDecision
from ..models.project import ProjectRegistry


class PolicyEngine:
    """Deterministic resource and resolved-path checks before any execution."""

    def __init__(self, config: PolicyConfig, registry: ProjectRegistry):
        self.config = config
        self.root = config.allowed_root.resolve()
        self.registry = registry

    @classmethod
    def load(cls, path: Path, root: Path, registry: ProjectRegistry) -> "PolicyEngine":
        config = PolicyConfig.model_validate(yaml.safe_load(path.read_text()))
        config.allowed_root = (root / config.allowed_root).resolve()
        return cls(config, registry)

    def safe_path(self, path: Path) -> Path:
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Path outside allowed_root: {path}")
        return resolved

    def validate(self, job: JobSpec, active_jobs: int = 0, retries: int = 0,
                 changed_fields: set[str] | None = None) -> PolicyDecision:
        violations: list[str] = []
        if job.queue is not None and job.queue not in self.config.allowed_queues:
            violations.append("queue is not whitelisted")
        for field in ("gpu_count", "cpu_count", "memory_gb", "timeout_minutes"):
            if getattr(job, field) > getattr(self.config, f"max_{field}"):
                violations.append(f"{field} exceeds policy limit")
        if active_jobs >= self.config.max_parallel_jobs:
            violations.append("max_parallel_jobs exceeded")
        if retries > self.config.max_retries:
            violations.append("max_retries exceeded")
        try:
            spec = self.registry.get(job.project_name)
            if job.entrypoint not in self.config.allowed_entrypoints or job.entrypoint not in (spec.train_entrypoint, spec.eval_entrypoint):
                violations.append("entrypoint is not registered and whitelisted")
            if job.working_directory.resolve() != spec.working_directory:
                violations.append("working_directory differs from registered project")
            for path in (job.working_directory, job.config_path, job.output_dir,
                         job.working_directory / job.entrypoint):
                self.safe_path(path)
        except (ValueError, OSError, RuntimeError) as exc:
            violations.append(str(exc))
        fields = changed_fields or set()
        requires = bool(fields & set(self.config.approval_required_fields))
        if changed_fields is not None:
            disallowed = fields - set(self.config.allowed_auto_fix_fields) - set(self.config.approval_required_fields)
            if disallowed:
                violations.append(f"Automatic changes forbidden: {sorted(disallowed)}")
        return PolicyDecision(allowed=not violations and not requires, violations=violations,
                              requires_approval=requires,
                              reason="; ".join(violations) or ("Approval required" if requires else "Allowed"))
