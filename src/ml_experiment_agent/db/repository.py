from typing import Any, TypeVar
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from .models import Base, Experiment, Run, Attempt, Metric, Artifact, AuditEvent

T = TypeVar("T", bound=Base)
ACTIVE = ("QUEUED", "RUNNING")


class Repository:
    """All queries live here; services commit each application operation."""

    def __init__(self, session: Session):
        self.session = session

    def add(self, record: T) -> T:
        self.session.add(record)
        self.session.flush()
        return record

    def get(self, model: type[T], identifier: Any) -> T:
        record = self.session.get(model, identifier)
        if record is None:
            raise KeyError(f"Unknown {model.__name__}: {identifier}")
        return record

    def list(self, model: type[T], **filters: Any) -> list[T]:
        return list(self.session.scalars(select(model).filter_by(**filters)))

    def delete(self, record: Base) -> None:
        self.session.delete(record)
        self.session.flush()

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()

    def latest_attempt(self, run_id: str) -> Attempt | None:
        return self.session.scalar(select(Attempt).where(Attempt.run_id == run_id)
                                   .order_by(Attempt.attempt_no.desc()).limit(1))

    def latest_audit(self, experiment_id: str, action: str, *, run_id: str | None = None,
                     attempt_id: str | None = None) -> AuditEvent | None:
        return self.session.scalar(select(AuditEvent).filter_by(
            experiment_id=experiment_id, action=action, run_id=run_id, attempt_id=attempt_id
        ).order_by(AuditEvent.id.desc()).limit(1))

    def active_count(self) -> int:
        return self.session.scalar(select(func.count()).select_from(Attempt).where(Attempt.status.in_(ACTIVE))) or 0

    def next_attempt_no(self, run_id: str) -> int:
        return (self.session.scalar(select(func.max(Attempt.attempt_no)).where(Attempt.run_id == run_id)) or 0) + 1

    def save_metric(self, **values: Any) -> Metric:
        rows = self.list(Metric, attempt_id=values["attempt_id"], metric_name=values["metric_name"], split=values["split"])
        if rows:
            rows[0].metric_value = values["metric_value"]
            return rows[0]
        return self.add(Metric(**values))

    def save_artifact(self, **values: Any) -> Artifact:
        rows = self.list(Artifact, attempt_id=values["attempt_id"], artifact_type=values["artifact_type"], path=values["path"])
        if rows:
            rows[0].metadata_json = values["metadata_json"]
            return rows[0]
        return self.add(Artifact(**values))

    def audit(self, experiment_id: str, action: str, *, run_id: str | None = None,
              attempt_id: str | None = None, outcome: str = "success",
              inputs: dict | None = None, outputs: dict | None = None) -> AuditEvent:
        return self.add(AuditEvent(experiment_id=experiment_id, run_id=run_id,
                                  attempt_id=attempt_id, actor="service", action=action,
                                  input_json=inputs or {}, output_json=outputs or {}, outcome=outcome))

    def sync_status(self, attempt: Attempt) -> None:
        run = self.get(Run, attempt.run_id)
        run.status = attempt.status
        self.session.flush()
        experiment = self.get(Experiment, run.experiment_id)
        statuses = [r.status for r in self.list(Run, experiment_id=experiment.id)]
        if all(s == "SUCCEEDED" for s in statuses):
            experiment.status = "SUCCEEDED"
        elif all(s in ("SUCCEEDED", "FAILED", "TIMED_OUT", "CANCELLED") for s in statuses):
            experiment.status = "FAILED"
        else:
            experiment.status = "RUNNING"
