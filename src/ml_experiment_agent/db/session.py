from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from .models import Base


def create_session_factory(path: Path):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path.as_posix()}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
