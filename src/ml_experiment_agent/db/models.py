from datetime import datetime, timezone
from uuid import uuid4
from typing import Any
from sqlalchemy import JSON, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def now() -> datetime:
    return datetime.now(timezone.utc)


def identifier() -> str:
    return uuid4().hex


class Base(DeclarativeBase):
    """SQLAlchemy declarative mapping root."""


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(primary_key=True, default=identifier)
    goal: Mapped[str]
    status: Mapped[str] = mapped_column(default="CREATED")
    plan_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=now)
    updated_at: Mapped[datetime] = mapped_column(default=now, onupdate=now)
    runs: Mapped[list["Run"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (UniqueConstraint("experiment_id", "name"),)
    id: Mapped[str] = mapped_column(primary_key=True, default=identifier)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"))
    name: Mapped[str]
    parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    base_config_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(default="CREATED")
    created_at: Mapped[datetime] = mapped_column(default=now)
    updated_at: Mapped[datetime] = mapped_column(default=now, onupdate=now)
    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    attempts: Mapped[list["Attempt"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Attempt(Base):
    __tablename__ = "attempts"
    __table_args__ = (UniqueConstraint("run_id", "attempt_no"),)
    id: Mapped[str] = mapped_column(primary_key=True, default=identifier)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    attempt_no: Mapped[int]
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    backend: Mapped[str] = mapped_column(default="mock")
    job_id: Mapped[str | None]
    status: Mapped[str] = mapped_column(default="CREATED")
    submitted_at: Mapped[datetime | None]
    started_at: Mapped[datetime | None]
    ended_at: Mapped[datetime | None]
    error_type: Mapped[str | None]
    error_message: Mapped[str | None]
    run: Mapped[Run] = relationship(back_populates="attempts")
    metrics: Mapped[list["Metric"]] = relationship(cascade="all, delete-orphan")
    artifacts: Mapped[list["Artifact"]] = relationship(cascade="all, delete-orphan")
    diagnoses: Mapped[list["Diagnosis"]] = relationship(cascade="all, delete-orphan")


class Metric(Base):
    __tablename__ = "metrics"
    __table_args__ = (UniqueConstraint("attempt_id", "metric_name", "split"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("attempts.id"))
    metric_name: Mapped[str]
    metric_value: Mapped[float]
    split: Mapped[str]


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("attempt_id", "artifact_type", "path"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("attempts.id"))
    artifact_type: Mapped[str]
    path: Mapped[str]
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class Diagnosis(Base):
    __tablename__ = "diagnoses"
    id: Mapped[int] = mapped_column(primary_key=True)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("attempts.id"))
    error_type: Mapped[str]
    summary: Mapped[str]
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    retrieved_doc_ids_json: Mapped[list[str]] = mapped_column(JSON)
    proposed_patch_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    decision: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=now)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"))
    attempt_id: Mapped[str | None] = mapped_column(ForeignKey("attempts.id"))
    actor: Mapped[str]
    action: Mapped[str]
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    output_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    outcome: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=now)
