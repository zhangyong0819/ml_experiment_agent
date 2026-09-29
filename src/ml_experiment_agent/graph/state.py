from typing import Any, TypedDict


class ExperimentGraphState(TypedDict, total=False):
    experiment_id: str
    user_goal: str
    pending_run_ids: list[str]
    ready_run_ids: list[str]
    active_run_ids: list[str]
    succeeded_run_ids: list[str]
    failed_run_ids: list[str]
    blocked_run_ids: list[str]
    diagnosed_run_ids: list[str]
    retry_source_attempt_ids: list[str]
    approval_attempt_ids: list[str]
    newly_succeeded_run_ids: list[str]
    newly_failed_run_ids: list[str]
    monitor_cycle: int
    max_monitor_cycles: int
    comparison: dict[str, Any] | None
    final_report: str | None
    report_path: str | None
    workflow_status: str
    stop_reason: str | None
