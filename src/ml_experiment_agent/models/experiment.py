from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import ArtifactType


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunConfig(StrictModel):
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    batch_size: int = Field(default=32, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    epochs: int = Field(default=3, gt=0)
    hidden_dim: int = Field(default=32, gt=0)
    weight_decay: float = Field(default=0, ge=0, allow_inf_nan=False)
    num_workers: int = Field(default=0, ge=0)
    precision: Literal["float32"] = "float32"
    output_dir: Path

    @field_validator("output_dir", mode="before")
    @classmethod
    def valid_path(cls, value: Any) -> Any:
        raw = str(value)
        if not raw.strip() or any(ord(c) < 32 for c in raw) or ".." in Path(raw).parts or ".." in PureWindowsPath(raw).parts:
            raise ValueError("output_dir must be a nonempty path without traversal")
        parts = PureWindowsPath(raw).parts
        anchor = PureWindowsPath(raw).anchor
        if any(any(c in part for c in '<>:"|?*') for part in parts if part != anchor):
            raise ValueError("output_dir contains invalid filesystem characters")
        return value


class RunPlan(StrictModel):
    name: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    base_config: RunConfig


class ExperimentDefinition(StrictModel):
    goal: str = Field(min_length=1)
    project_name: str = "toy_classification"
    runs: list[RunPlan] = Field(default_factory=list)


class ArtifactRecord(StrictModel):
    attempt_id: str
    artifact_type: ArtifactType
    path: Path
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricRecord(StrictModel):
    attempt_id: str
    metric_name: Literal["accuracy", "macro_f1", "ece"]
    metric_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    split: Literal["test"] = "test"
