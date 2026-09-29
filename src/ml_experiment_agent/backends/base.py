from abc import ABC, abstractmethod
from ..models.job import JobSpec, JobHandle, JobLog
from ..models.enums import JobStatus


class ComputeBackend(ABC):
    @abstractmethod
    def submit(self, job: JobSpec) -> JobHandle:
        """Submit a registered project with validated configuration."""

    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        """Read the backend job state."""

    @abstractmethod
    def logs(self, job_id: str, tail: int = 200) -> JobLog:
        """Read at most tail log lines."""

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Cancel an active job; terminal jobs remain unchanged."""
