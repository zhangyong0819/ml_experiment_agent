from pathlib import Path
from pydantic import Field
from .experiment import StrictModel
from .enums import JobStatus, Scenario


class JobSpec(StrictModel):
    queue: str | None = None
    project_name: str
    entrypoint: str
    config_path: Path
    working_directory: Path
    output_dir: Path
    gpu_count: int = Field(default=0, ge=0)
    cpu_count: int = Field(default=1, gt=0)
    memory_gb: float = Field(default=2, gt=0, allow_inf_nan=False)
    timeout_minutes: float = Field(default=5, gt=0, allow_inf_nan=False)
    scenario: Scenario = Scenario.SUCCESS


class JobHandle(StrictModel):
    job_id: str
    backend: str = "mock"
    status: JobStatus = JobStatus.QUEUED


class JobLog(StrictModel):
    job_id: str
    text: str
