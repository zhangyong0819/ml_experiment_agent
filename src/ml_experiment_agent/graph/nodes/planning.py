from langchain_core.runnables import RunnableConfig
from ...db.models import Experiment
from ...models.experiment import ExperimentDefinition
from ..dependencies import WorkflowDependencies
from ..state import ExperimentGraphState
from .common import audit_once, snapshot


def plan_experiment(state: ExperimentGraphState, config: RunnableConfig,
                    deps: WorkflowDependencies) -> ExperimentGraphState:
    experiment = deps.repository.get(Experiment, state["experiment_id"])
    if str(config.get("configurable", {}).get("thread_id", "")) != experiment.id:
        raise ValueError("thread_id must equal experiment_id")
    if state.get("user_goal", experiment.goal) != experiment.goal:
        raise ValueError("Workflow goal differs from application database")
    maximum = state.get("max_monitor_cycles", 20)
    cycle = state.get("monitor_cycle", 0)
    if type(maximum) is not int or not 1 <= maximum <= 1000 or type(cycle) is not int or cycle < 0:
        raise ValueError("Monitor bounds must be integers: max 1..1000, cycle >= 0")
    if deps.repository.latest_audit(experiment.id, "plan_created") is None:
        spec = deps.experiment_service.registry.get(experiment.plan_json["project_name"])
        plan = deps.planner.plan(experiment.goal, spec)
        plan = ExperimentDefinition.model_validate(plan.model_dump(mode="json"))
        if plan.project_name != spec.project_name or plan.goal != experiment.goal:
            raise ValueError("Planner changed experiment identity or goal")
        names = [run.name for run in plan.runs]
        if not names or len(names) != len(set(names)):
            raise ValueError("Planner must provide nonempty, unique run identities")
        experiment.plan_json = plan.model_dump(mode="json")
        audit_once(deps, experiment.id, "plan_created", outputs={"run_names": names})
    return snapshot(deps, state) | {"user_goal": experiment.goal, "ready_run_ids": [],
        "newly_succeeded_run_ids": [], "newly_failed_run_ids": [], "monitor_cycle": cycle,
        "max_monitor_cycles": maximum, "comparison": None, "final_report": None,
        "report_path": None, "workflow_status": "RUNNING", "stop_reason": None}
