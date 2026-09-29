from ...services.comparison_service import ComparisonService
from ...services.report_service import ReportService
from ..dependencies import WorkflowDependencies
from ..state import ExperimentGraphState
from .common import audit_once


def compare_runs(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    comparison = ComparisonService(deps.repository).compare(state["experiment_id"])
    result = comparison.model_dump(mode="json")
    audit_once(deps, state["experiment_id"], "comparison_completed", outputs=result)
    return {"comparison": result}


def generate_report(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    # Re-read application facts instead of trusting an old checkpoint's comparison copy.
    comparison = ComparisonService(deps.repository).compare(state["experiment_id"])
    service = ReportService(deps.repository, deps.policy_engine)
    report = service.render(comparison, state.get("stop_reason"))
    path = service.save(state["experiment_id"], report)
    audit_once(deps, state["experiment_id"], "report_generated", outputs={"path": str(path), "comparison_status": comparison.status})
    return {"comparison": comparison.model_dump(mode="json"), "final_report": report, "report_path": str(path)}
