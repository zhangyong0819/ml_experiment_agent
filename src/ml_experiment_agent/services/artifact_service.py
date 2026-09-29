from pathlib import Path
from ..db.repository import Repository
from ..models.experiment import ArtifactRecord
from .policy_engine import PolicyEngine


class ArtifactService:
    def __init__(self, repository: Repository, policy: PolicyEngine):
        self.repository = repository
        self.policy = policy

    def record(self, record: ArtifactRecord, output_dir: Path) -> None:
        path = self.policy.safe_path(record.path)
        if not path.is_relative_to(self.policy.safe_path(output_dir)) or not path.is_file():
            raise ValueError(f"Artifact {record.artifact_type.value} must be an existing file inside attempt output_dir: {path}")
        self.repository.save_artifact(attempt_id=record.attempt_id, artifact_type=record.artifact_type.value,
                                      path=str(path), metadata_json=record.metadata)
