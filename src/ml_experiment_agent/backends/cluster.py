"""Local operator configuration, never supplied by the planner or API request."""
from pathlib import Path, PurePosixPath
import re
import os
from typing import Literal
import yaml
from pydantic import Field, field_validator, model_validator
from ..models.experiment import StrictModel

QUEUES = {"734090d!", "944090d!", "9654p6000ib!"}


def remote_path(value: str) -> str:
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", value) or ".." in PurePosixPath(value).parts:
        raise ValueError("Remote path must be absolute and contain no traversal or shell syntax")
    normalized = str(PurePosixPath(value))
    if normalized == "/" or value.startswith("//"):
        raise ValueError("Remote root cannot be the filesystem root")
    return normalized


class ClusterConfig(StrictModel):
    enabled: bool = False
    ssh_host: str | None = None
    ssh_host_alias: str | None = None
    ssh_port: int = Field(default=22, ge=1, le=65535, strict=True)
    ssh_username: str | None = None
    interactive_auth: bool = False
    remote_project_root: str
    memory_unit: Literal["MB", "GB"] | None = None
    allowed_queues: list[str] = Field(default_factory=lambda: sorted(QUEUES))
    queue: str = "734090d!"
    poll_interval_seconds: float = Field(default=30, ge=5, le=3600)

    @field_validator("ssh_host", "ssh_host_alias")
    @classmethod
    def host_alias(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
            raise ValueError("Invalid SSH hostname or alias")
        return value

    @field_validator("ssh_username")
    @classmethod
    def username(cls, value: str | None) -> str | None:
        # Slash is an opaque username character, not a delimiter or a path.
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./@-]{0,255}", value):
            raise ValueError("Invalid SSH username")
        return value

    @field_validator("remote_project_root")
    @classmethod
    def root_path(cls, value: str) -> str:
        return remote_path(value)

    @model_validator(mode="after")
    def queues(self):
        if bool(self.ssh_host_alias) == bool(self.ssh_host):
            raise ValueError("Choose direct ssh_host or ssh_host_alias")
        if self.ssh_host and not self.ssh_username:
            raise ValueError("Direct SSH requires ssh_username")
        if self.ssh_host_alias and self.ssh_username:
            raise ValueError("Alias mode obtains its username from SSH config")
        if not set(self.allowed_queues) <= QUEUES or self.queue not in self.allowed_queues:
            raise ValueError("Only the three confirmed school queues are supported")
        return self

    def ssh_arguments(self) -> list[str]:
        """OpenSSH argv; the username is passed verbatim as one -l argument."""
        args = ["ssh", "-o", "BatchMode=no" if self.interactive_auth else "BatchMode=yes",
                "-o", "ConnectTimeout=10", "-o",
                "StrictHostKeyChecking=yes"]
        if self.ssh_host_alias:
            return args + [self.ssh_host_alias]
        return args + ["-p", str(self.ssh_port), "-l", self.ssh_username, self.ssh_host]

    @property
    def connection_identity(self) -> dict:
        return {"host": self.ssh_host, "alias": self.ssh_host_alias,
                "port": self.ssh_port, "username": self.ssh_username}

    @classmethod
    def from_environment(cls) -> "ClusterConfig":
        """Read operator-supplied process variables; never load or persist secrets."""
        return cls.model_validate({
            "enabled": os.getenv("ENABLE_REAL_LSF", "false").lower() == "true",
            "ssh_host": os.getenv("LSF_SSH_HOST") or None,
            "ssh_host_alias": os.getenv("LSF_SSH_HOST_ALIAS") or None,
            "ssh_port": int(os.getenv("LSF_SSH_PORT") or "22"),
            "ssh_username": os.getenv("LSF_SSH_USERNAME") or None,
            "remote_project_root": os.getenv("LSF_REMOTE_PROJECT_ROOT", ""),
            "interactive_auth": os.getenv("LSF_SSH_INTERACTIVE", "false").lower() == "true",
            "memory_unit": os.getenv("LSF_MEMORY_UNIT") or None,
            "queue": os.getenv("LSF_QUEUE") or "734090d!",
        })

    @classmethod
    def load(cls, path: Path) -> "ClusterConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))["cluster"])
