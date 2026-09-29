from pathlib import Path
from pydantic import Field
import yaml
from .experiment import StrictModel


class ProjectSpec(StrictModel):
    project_name: str
    working_directory: Path
    train_entrypoint: str
    eval_entrypoint: str
    modifiable_config_fields: list[str]
    checkpoint_name: str = "checkpoint.pt"
    metadata_name: str = "metadata.json"
    evaluation_name: str = "metrics.json"
    metric_names: list[str] = Field(default_factory=lambda: ["accuracy", "macro_f1", "ece"])


class ProjectRegistry:
    """Load trusted registrations; jobs can only reference these projects."""

    def __init__(self, directory: Path, root: Path):
        self.projects: dict[str, ProjectSpec] = {}
        for path in sorted(directory.glob("*.yaml")):
            spec = ProjectSpec.model_validate(yaml.safe_load(path.read_text()))
            spec.working_directory = (root / spec.working_directory).resolve()
            if spec.project_name in self.projects:
                raise ValueError("Duplicate project registration")
            self.projects[spec.project_name] = spec

    def get(self, name: str) -> ProjectSpec:
        if name not in self.projects:
            raise ValueError(f"Unregistered project: {name}")
        return self.projects[name].model_copy(deep=True)
