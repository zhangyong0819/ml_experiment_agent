from typing import Any, Literal
from pydantic import Field
from .experiment import StrictModel


class RunSummary(StrictModel):
    run_id: str
    name: str
    status: str
    attempt_id: str | None = None
    attempt_status: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    error_type: str | None = None
    reason: str | None = None


class MetricWinner(StrictModel):
    direction: Literal["highest", "lowest"]
    value: float | None = None
    run_ids: list[str] = Field(default_factory=list)


class ExperimentComparison(StrictModel):
    experiment_id: str
    status: Literal["COMPLETE", "PARTIAL", "NO_SUCCESS"]
    runs: list[RunSummary]
    successful_runs: list[str]
    failed_runs: list[str]
    blocked_runs: list[str]
    incomplete_runs: list[str]
    metric_results: dict[str, MetricWinner]
