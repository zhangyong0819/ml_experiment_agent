from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator
from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def sqlite_checkpointer(path: Path, *, application_db: Path) -> Iterator[SqliteSaver]:
    """Own the official saver's lifetime; keep graph tables out of the application DB."""
    path = path.resolve()
    if path == application_db.resolve():
        raise ValueError("Checkpoint and application databases must be separate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        saver.setup()
        yield saver
