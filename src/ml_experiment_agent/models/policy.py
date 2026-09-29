from pathlib import Path
from pydantic import Field
from .experiment import StrictModel


class PolicyConfig(StrictModel):
    allowed_queues: list[str] = Field(default_factory=lambda: ["734090d!", "944090d!", "9654p6000ib!"])
    allowed_root: Path
    allowed_entrypoints: list[str]
    max_gpu_count: int = Field(default=1, ge=0)
    max_cpu_count: int = Field(default=8, gt=0)
    max_memory_gb: float = Field(default=32, gt=0)
    max_timeout_minutes: float = Field(default=60, gt=0)
    max_parallel_jobs: int = Field(default=3, gt=0)
    max_retries: int = Field(default=2, ge=0)
    allowed_auto_fix_fields: list[str] = Field(default_factory=list)
    approval_required_fields: list[str] = Field(default_factory=list)


class PolicyDecision(StrictModel):
    allowed: bool
    violations: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    reason: str
