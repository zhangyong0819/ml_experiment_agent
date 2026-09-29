from pathlib import Path
from typing import Any
import yaml
from ..models.experiment import RunConfig
from ..models.project import ProjectSpec


class ConfigService:
    def load(self, path: Path) -> RunConfig:
        return RunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def generate(self, base: RunConfig, parameters: dict[str, Any], project: ProjectSpec) -> RunConfig:
        unknown = set(parameters) - set(project.modifiable_config_fields)
        if unknown:
            raise ValueError(f"Nonmodifiable config fields: {sorted(unknown)}")
        return RunConfig.model_validate(base.model_dump() | parameters)

    def save(self, config: RunConfig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
