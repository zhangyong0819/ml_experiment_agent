import json
import subprocess
import sys
from pathlib import Path
from pydantic import Field
from ..models.experiment import StrictModel, MetricRecord
from ..models.job import JobSpec
from ..models.project import ProjectSpec
from ..db.repository import Repository
from .config_service import ConfigService
from .policy_engine import PolicyEngine


class EvaluationMetrics(StrictModel):
    accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_f1: float = Field(ge=0, le=1, allow_inf_nan=False)
    ece: float = Field(ge=0, le=1, allow_inf_nan=False)


class EvaluationService:
    """Materialize real CPU toy artifacts after mock success, then persist metrics."""

    def __init__(self, repository: Repository, policy: PolicyEngine):
        self.repository = repository
        self.policy = policy

    def _invoke(self, spec: ProjectSpec, entrypoint: str, config_path: Path) -> None:
        config = ConfigService().load(config_path)
        job = JobSpec(project_name=spec.project_name, entrypoint=entrypoint, config_path=config_path,
                      working_directory=spec.working_directory, output_dir=config.output_dir,
                      cpu_count=config.num_workers + 1)
        decision = self.policy.validate(job)
        if not decision.allowed:
            raise ValueError(decision.reason)
        subprocess.run([sys.executable, str(self.policy.safe_path(spec.working_directory / entrypoint)),
                        "--config", str(self.policy.safe_path(config_path))], cwd=spec.working_directory,
                       check=True, capture_output=True, text=True, timeout=job.timeout_minutes * 60)

    def prepare_artifacts(self, spec: ProjectSpec, config_path: Path) -> None:
        """Run real toy training explicitly, independently of evaluation."""
        self._check_output_paths(spec, config_path)
        self._invoke(spec, spec.train_entrypoint, config_path)

    def _check_output_paths(self, spec: ProjectSpec, config_path: Path) -> Path:
        config = ConfigService().load(self.policy.safe_path(config_path))
        output = self.policy.safe_path(config.output_dir)
        # Revalidate all convention paths before child processes can write them.
        for name in (spec.checkpoint_name, spec.metadata_name, spec.evaluation_name):
            path = self.policy.safe_path(output / name)
            if not path.is_relative_to(output):
                raise ValueError("Artifact convention escapes output directory")
        return output

    def evaluate(self, attempt_id: str, spec: ProjectSpec, config_path: Path) -> list[MetricRecord]:
        """Phase 1 compatible convenience: train, then evaluate."""
        self.prepare_artifacts(spec, config_path)
        return self.evaluate_existing(attempt_id, spec, config_path)

    def evaluate_existing(self, attempt_id: str, spec: ProjectSpec, config_path: Path) -> list[MetricRecord]:
        """Evaluate an existing checkpoint; never repair missing training output."""
        output = self._check_output_paths(spec, config_path)
        if not (output / spec.checkpoint_name).is_file():
            raise FileNotFoundError(f"Missing checkpoint: {output / spec.checkpoint_name}")
        self._invoke(spec, spec.eval_entrypoint, config_path)
        values = EvaluationMetrics.model_validate(json.loads((output / spec.evaluation_name).read_text())).model_dump()
        if set(values) != set(spec.metric_names):
            raise ValueError("Evaluation metrics differ from registered metric schema")
        records = [MetricRecord(attempt_id=attempt_id, metric_name=name, metric_value=value)
                   for name, value in values.items()]
        for record in records:
            self.repository.save_metric(**record.model_dump())
        return records
