import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    database_path: Path

    @classmethod
    def from_root(cls, root: Path) -> "Settings":
        root = root.resolve()
        database = (root / os.getenv("ML_AGENT_DB", "data/experiments.sqlite")).resolve()
        if not database.is_relative_to(root):
            raise ValueError("Database must be inside project root")
        return cls(root, database)
